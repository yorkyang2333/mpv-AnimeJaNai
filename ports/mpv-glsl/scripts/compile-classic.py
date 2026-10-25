"""Ahead-of-time mpv GLSL lowering for Real-ESRGAN Compact ONNX models."""
from pathlib import Path
import argparse
import numpy as np
import onnx
from onnx import numpy_helper,helper


def f(x):
    s=format(float(x),'.9g')
    return s if '.' in s or 'e' in s else s+'.0'


def vector(x):
    return 'vec4('+', '.join(f(v) for v in x)+')'


def compile_model(path,prefix):
    model=onnx.load(path)
    weights={w.name:numpy_helper.to_array(w) for w in model.graph.initializer}
    nodes=list(model.graph.node)
    parts=[];inputs=['MAIN'];inc=3;index=0;layer=0
    while index<len(nodes) and nodes[index].op_type=='Conv':
        n=nodes[index];w=weights[n.input[1]];bias=weights[n.input[2]]
        oc,ic,kh,kw=w.shape
        attrs={a.name:helper.get_attribute_value(a) for a in n.attribute}
        assert ic==inc and kh==kw==3 and attrs['group']==1 and attrs['pads']==[1]*4 and attrs['strides']==[1,1] and attrs['dilations']==[1,1]
        index+=1;slope=None
        if index<len(nodes) and nodes[index].op_type=='PRelu':
            assert nodes[index].input[0]==n.output[0]
            slope=weights[nodes[index].input[1]].reshape(-1)
            assert slope.shape==(oc,)
            index+=1
        outputs=[]
        for og in range(oc//4):
            name=f'{prefix}L{layer}G{og}';outputs.append(name)
            hdr=[f'//!DESC {prefix} convolution {layer}, channels {og*4}-{og*4+3}', '//!HOOK MAIN']
            hdr += [f'//!BIND {t}' for t in inputs]
            hdr += [f'//!SAVE {name}','//!WIDTH MAIN.w','//!HEIGHT MAIN.h','//!COMPONENTS 4']
            body=['vec4 hook() {',f'    vec2 sz = {inputs[0]}_size;',f'    vec2 p = floor({inputs[0]}_pos * sz);',f'    vec4 r = {vector(bias[og*4:og*4+4])};']
            for y in range(3):
                for x in range(3):
                    body.append(f'    vec2 q{x}{y} = p + vec2({x-1}.0, {y-1}.0);')
                    body.append(f'    if (all(greaterThanEqual(q{x}{y}, vec2(0.0))) && all(lessThan(q{x}{y}, sz))) {{')
                    for ig,t in enumerate(inputs):
                        matrix=np.zeros((4,4),np.float32)
                        count=min(4,ic-ig*4)
                        matrix[:,:count]=w[og*4:og*4+4,ig*4:ig*4+count,y,x]
                        literal='mat4('+', '.join(f(v) for v in matrix.T.reshape(-1))+')'
                        body.append(f'        r += {literal} * {t}_tex((q{x}{y} + vec2(0.5)) / sz);')
                    body.append('    }')
            if slope is not None:
                body.append(f'    r = max(r, vec4(0.0)) + min(r, vec4(0.0)) * {vector(slope[og*4:og*4+4])};')
            body += ['    return r;','}']
            parts.append('\n'.join(hdr+body)+'\n')
        inputs=outputs;inc=oc;layer+=1
    assert [n.op_type for n in nodes[index:]]==['DepthToSpace','Resize','Add','Clip']
    attrs={a.name:helper.get_attribute_value(a) for a in nodes[index].attribute}
    assert attrs=={'blocksize':2,'mode':b'CRD'} and inc==12
    resize={a.name:helper.get_attribute_value(a) for a in nodes[index+1].attribute}
    assert resize['mode']==b'nearest' and resize['coordinate_transformation_mode']==b'asymmetric'
    assert np.array_equal(weights[nodes[index+1].input[2]],np.array([1,1,2,2]))
    assert float(weights[nodes[index+3].input[1]])==0 and float(weights[nodes[index+3].input[2]])==1
    hdr=[f'//!DESC {prefix} pixel shuffle, nearest residual, clip', '//!HOOK MAIN','//!BIND MAIN']+[f'//!BIND {t}' for t in inputs]+['//!WIDTH MAIN.w 2 *','//!HEIGHT MAIN.h 2 *','//!COMPONENTS 4']
    body=['vec4 hook() {','    vec2 q = floor(MAIN_pos * MAIN_size * 2.0);','    vec2 p = floor(q / 2.0);','    ivec2 phase = ivec2(mod(q, 2.0));','    int k = phase.y * 2 + phase.x;',f'    vec3 result = vec3({inputs[0]}_tex((p + vec2(0.5)) / MAIN_size)[k], {inputs[1]}_tex((p + vec2(0.5)) / MAIN_size)[k], {inputs[2]}_tex((p + vec2(0.5)) / MAIN_size)[k]);','    result += MAIN_tex((p + vec2(0.5)) / MAIN_size).rgb;','    return vec4(clamp(result, 0.0, 1.0), 1.0);','}']
    parts.append('\n'.join(hdr+body)+'\n')
    return '// Original AnimeJaNai weights by the-database, CC BY-NC-SA 4.0. PReLU fused into Conv.\n'+'\n'.join(parts)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model',type=Path);parser.add_argument('output',type=Path);parser.add_argument('--prefix',default='AJNSUC')
    args=parser.parse_args();args.output.write_text(compile_model(args.model,args.prefix))
