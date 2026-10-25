// Area downsampling before neural inference; reduces detail as well as cost.
//!DESC AnimeJaNai inference limit 240p, area x
//!HOOK MAIN
//!BIND MAIN
//!WIDTH MAIN.w 240 MAIN.h / *
//!HEIGHT MAIN.h
//!WHEN MAIN.h 240 >
//!COMPONENTS 4
vec4 hook() {
    float ratio = MAIN_size.y / 240.0;
    float center = MAIN_pos.x * MAIN_size.x;
    float left = max(0.0, center - ratio * 0.5);
    float right = min(MAIN_size.x, center + ratio * 0.5);
    vec4 result = vec4(0.0);
    float total = 0.0;
    for (int i = int(floor(left)); i < int(ceil(right)); i++) {
        float weight = max(0.0, min(right, float(i) + 1.0) - max(left, float(i)));
        result += MAIN_tex(vec2((float(i) + 0.5) / MAIN_size.x, MAIN_pos.y)) * weight;
        total += weight;
    }
    return result / max(total, 0.000001);
}

//!DESC AnimeJaNai inference limit 240p, area y
//!HOOK MAIN
//!BIND MAIN
//!WIDTH MAIN.w
//!HEIGHT 240
//!WHEN MAIN.h 240 >
//!COMPONENTS 4
vec4 hook() {
    float ratio = MAIN_size.y / 240.0;
    float center = MAIN_pos.y * MAIN_size.y;
    float left = max(0.0, center - ratio * 0.5);
    float right = min(MAIN_size.y, center + ratio * 0.5);
    vec4 result = vec4(0.0);
    float total = 0.0;
    for (int i = int(floor(left)); i < int(ceil(right)); i++) {
        float weight = max(0.0, min(right, float(i) + 1.0) - max(left, float(i)));
        result += MAIN_tex(vec2(MAIN_pos.x, (float(i) + 0.5) / MAIN_size.y)) * weight;
        total += weight;
    }
    return result / max(total, 0.000001);
}
