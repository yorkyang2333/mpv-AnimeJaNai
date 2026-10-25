#!/usr/bin/env python3
"""Compile the pinned AnimeJaNai SPANF3 ONNX graph to mpv fragment passes.

Channel groups occupy RGBA tiles. Model weights are immutable float textures;
this keeps shader source and sampler counts within macOS OpenGL limits.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT.parents[1] / 'BuildMpvUpscale2xAnimeJaNai/mpv-upscale-2x_animejanai/animejanai/onnx/2x_AnimeJaNai_HD_V3.1_Performance_SPANF3_b5f48_unshuffle_fp16.onnx'
OUTPUT = ROOT / 'shaders/AnimeJaNai_Performance.glsl'


def layout(channels):
    groups = math.ceil(channels / 4)
    cols = math.ceil(math.sqrt(groups))
    return cols, math.ceil(groups / cols)


def number(value):
    s = format(float(value), '.9g')
    return s if '.' in s or 'e' in s else s + '.0'


class Compiler:
    def __init__(self):
        self.blocks = []
        self.tensors = {}
        self.counter = 0

    def tensor(self, key, channels):
        self.counter += 1
        value = (f'AJN{self.counter}', channels)
        self.tensors[key] = value
        return value

    def sampler(self, name, channels):
        cols, rows = layout(channels)
        return f'''
vec2 {name}_cell() {{ return {name}_size / vec2({cols}.0, {rows}.0); }}
vec4 read_{name}(vec2 p, int g) {{
    vec2 sz = {name}_cell();
    if (p.x < 0.0 || p.y < 0.0 || p.x >= sz.x || p.y >= sz.y) return vec4(0.0);
    vec2 tile = vec2(g % {cols}, g / {cols});
    return {name}_tex((tile * sz + p + vec2(0.5)) / {name}_size);
}}
'''

    def emit(self, output, binds, body, description, final=False):
        name, channels = output
        cols, rows = layout(channels)
        header = [f'//!DESC AnimeJaNai {description}', '//!HOOK MAIN']
        header += [f'//!BIND {n}' for n in dict.fromkeys(['MAIN'] + [b[0] for b in binds])]
        header += [f'//!SAVE {"MAIN" if final else name}',
                   f'//!WIDTH MAIN.w 2 *' if final else f'//!WIDTH MAIN.w 2 % MAIN.w + 2 / {cols} *',
                   f'//!HEIGHT MAIN.h 2 *' if final else f'//!HEIGHT MAIN.h 2 % MAIN.h + 2 / {rows} *',
                   '//!COMPONENTS 4']
        samplers = ''.join(self.sampler(n, c) for n, c in dict(binds).items() if c)
        position = '' if final else f'''
    vec2 sz = floor((MAIN_size + vec2(1.0)) / 2.0);
    vec2 at = floor(MAIN_pos * sz * vec2({cols}.0, {rows}.0));
    ivec2 tile = ivec2(floor(at / sz));
    int g = tile.y * {cols} + tile.x;
    vec2 p = mod(at, sz);
    if (g >= {math.ceil(channels / 4)}) return vec4(0.0);
'''
        self.blocks.append('\n'.join(header) + '\n' + samplers + '\nvec4 hook() {\n' + position + body + '\n}\n')

    def texture(self, name, data):
        height, width, _ = data.shape
        self.blocks.append(f'//!TEXTURE {name}\n//!SIZE {width} {height}\n//!FORMAT rgba32f\n//!FILTER NEAREST\n//!BORDER CLAMP\n' + data.astype('<f4').tobytes().hex() + '\n')

    def conv(self, node, weights):
        src, inc = self.tensors[node.input[0]]
        w = weights[node.input[1]].astype(np.float32)
        bias = weights[node.input[2]].astype(np.float32)
        outc, _, kh, kw = w.shape
        attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        assert attrs['group'] == 1 and attrs['strides'] == [1, 1] and attrs['dilations'] == [1, 1]
        assert attrs['pads'] == [kh // 2, kw // 2, kh // 2, kw // 2]
        assert inc % 4 == 0 and outc % 4 == 0
        output = self.tensor(node.output[0], outc)
        wn, bn = output[0] + 'W', output[0] + 'B'
        # Each texel holds four output coefficients for one input channel.
        data = w.reshape(outc // 4, 4, inc, kh, kw).transpose(0, 2, 3, 4, 1).reshape(outc // 4, inc * kh * kw, 4)
        self.texture(wn, data)
        self.texture(bn, bias.reshape(1, outc // 4, 4))
        body = f'''    vec4 r = texture({bn}, vec2((float(g) + 0.5) / {outc // 4}.0, 0.5));
    for (int ig = 0; ig < {inc // 4}; ig++) {{
        for (int y = 0; y < {kh}; y++) {{
            for (int x = 0; x < {kw}; x++) {{
                vec4 v = read_{src}(p + vec2(x - {kw // 2}, y - {kh // 2}), ig);
                int base = (ig * 4 * {kh} + y) * {kw} + x;
                vec2 uv = vec2((float(base) + 0.5) / {inc * kh * kw}.0, (float(g) + 0.5) / {outc // 4}.0);
                vec2 stepw = vec2({number(kh * kw / (inc * kh * kw))}, 0.0);
                r += mat4(texture({wn}, uv), texture({wn}, uv + stepw), texture({wn}, uv + stepw * 2.0), texture({wn}, uv + stepw * 3.0)) * v;
            }}
        }}
    }}
    return r;'''
        self.emit(output, [(src, inc), (wn, 0), (bn, 0)], body, node.name or output[0])

    def compile(self, model):
        weights = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}
        convs = [n for n in model.graph.node if n.op_type == 'Conv']
        first = convs[0]
        # This audited prefix reflects odd edges, then performs pixel-unshuffle(2).
        # Refuse different graphs instead of guessing preprocessing semantics.
        assert weights[first.input[1]].shape[1] == 12
        source = self.tensor(first.input[0], 12)
        self.emit(source, [], '''    vec4 r = vec4(0.0);
    for (int k = 0; k < 4; k++) {
        int c = g * 4 + k;
        vec2 q = p * 2.0 + vec2((c % 4) % 2, (c % 4) / 2);
        q = min(q, 2.0 * MAIN_size - vec2(2.0) - q);
        r[k] = MAIN_tex((q + vec2(0.5)) / MAIN_size)[c / 4];
    }
    return r;''', 'reflect pad and pixel unshuffle')
        started = False
        for node in model.graph.node:
            if node is first:
                started = True
            if not started:
                continue
            op = node.op_type
            if op == 'Conv':
                self.conv(node, weights)
            elif op in ('Relu', 'Tanh', 'Add', 'Mul'):
                inputs = [self.tensors[n] for n in node.input]
                output = self.tensor(node.output[0], inputs[0][1])
                expressions = [f'read_{n}(p, g)' for n, _ in inputs]
                if op == 'Relu': expr = f'max({expressions[0]}, vec4(0.0))'
                elif op == 'Tanh': expr = f'tanh({expressions[0]})'
                else: expr = (' + ' if op == 'Add' else ' * ').join(expressions)
                self.emit(output, inputs, f'    return {expr};', op)
            elif op == 'Concat':
                assert helper.get_attribute_value(node.attribute[0]) == 1
                inputs = [self.tensors[n] for n in node.input]
                output = self.tensor(node.output[0], sum(c for _, c in inputs))
                body = ''
                offset = 0
                for n, c in inputs:
                    body += f'    if (g < {offset + c // 4}) return read_{n}(p, g - {offset});\n'
                    offset += c // 4
                self.emit(output, inputs, body + '    return vec4(0.0);', 'channel concatenation')
            elif op == 'DepthToSpace':
                attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
                assert attrs == {'blocksize': 4, 'mode': b'CRD'}
                src, c = self.tensors[node.input[0]]
                assert c == 48
                body = f'''    vec2 q = floor(MAIN_pos * MAIN_size * 2.0);
    vec2 p = floor(q / 4.0);
    ivec2 phase = ivec2(mod(q, 4.0));
    int k = phase.y * 4 + phase.x;
    vec3 r;
    for (int c = 0; c < 3; c++) {{
        int ch = c * 16 + k;
        r[c] = read_{src}(p, ch / 4)[ch % 4];
    }}
    return vec4(r, 1.0);'''
                self.emit(('MAIN', 3), [(src, c)], body, 'pixel shuffle and crop (2x)', final=True)
                break
            else:
                raise ValueError(f'Unsupported feature node: {op}')
        return '\n'.join(self.blocks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=MODEL)
    parser.add_argument('--metadata', type=Path, default=ROOT / 'models/animejanai/model.json')
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    metadata = json.loads(args.metadata.read_text())
    digest = hashlib.sha256(args.model.read_bytes()).hexdigest()
    if digest != metadata['sha256']:
        raise ValueError('AnimeJaNai model checksum mismatch; this compiler requires the audited graph')
    compiler = Compiler()
    result = compiler.compile(onnx.load(args.model))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text('// Derived from AnimeJaNai by the-database; CC BY-NC-SA 4.0. See models/animejanai/.\n' + result)
    print(f'Compiled AnimeJaNai: {len(compiler.tensors)} tensors, {args.output.stat().st_size:,} bytes')


if __name__ == '__main__':
    main()
