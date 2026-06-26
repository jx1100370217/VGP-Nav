"""生成多数据集总入口 outputs/index.html。

扫描 outputs/ 下已建好(有 web/data.js)的数据集, 生成一个可切换地图的总入口页:
iframe 嵌各数据集的导航页, 顶部按钮切换地图。**可扩展: 新数据集建好后重跑本脚本即可。**

  /home/ubuntu/miniconda3/envs/internvla/bin/python scripts/gen_portal.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vgpnav.config import _PROJ
from vgpnav.datasets import DATASETS

OUT = os.path.join(_PROJ, "outputs")

# 友好显示名(没有则用数据集名)。加数据集时可在此补一行中文名。
LABELS = {
    "Mapping_C8": "深港国际C8",
    "ChuangfuTower": "创富大厦28楼",
    "Mappingdata_C7": "深港国际C7",
    "Mappingdata_Firstfloor": "深港国际1楼",
}

maps = []
for name in DATASETS:                       # 按注册顺序
    if not os.path.exists(os.path.join(OUT, name, "web", "data.js")):
        continue                            # 只列已建好的
    info = ""
    meta_p = os.path.join(OUT, name, "db", "meta.json")
    if os.path.exists(meta_p):
        m = json.load(open(meta_p))
        info = f"{m.get('n_traj','?')}帧·{m.get('n_db','?')}DB"
    maps.append([name, LABELS.get(name, name), info])

HTML = '''<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>VGP-Nav · 多地图导航</title>
<style>
*{box-sizing:border-box}
html,body{margin:0;height:100%;font-family:-apple-system,system-ui,"Microsoft YaHei",sans-serif;background:#070b14}
#bar{position:fixed;top:0;left:0;right:0;height:46px;background:linear-gradient(#0e1626,#0b1220);
  display:flex;align-items:center;gap:10px;padding:0 18px;z-index:999;border-bottom:1px solid #1c2942;box-shadow:0 2px 12px #0008}
.brand{color:#2de2e6;font-weight:700;font-size:15px;letter-spacing:.5px;white-space:nowrap}
.brand small{color:#5a6b8a;font-weight:400;font-size:11px;margin-left:6px}
.sep{color:#3a4a6a;font-size:12px;margin:0 2px;white-space:nowrap}
#tabs{display:flex;gap:8px;overflow-x:auto}
.tab{background:#15203a;color:#9fb3d1;border:1px solid #26365a;padding:7px 15px;border-radius:7px;
  cursor:pointer;font-size:13px;transition:.15s;white-space:nowrap}
.tab:hover{background:#1c2a48;color:#cfe0f5}
.tab.active{background:#2de2e6;color:#04121f;font-weight:700;border-color:#2de2e6}
.tab small{opacity:.7;font-size:11px;margin-left:5px}
iframe{position:fixed;top:46px;left:0;right:0;bottom:0;width:100%;height:calc(100% - 46px);border:0;background:#070b14}
#empty{color:#6b7a99;text-align:center;margin-top:120px;font-size:14px}
</style></head><body>
<div id="bar">
  <span class="brand">VGP·NAV<small>多地图导航</small></span>
  <span class="sep">▏ 选择地图:</span>
  <div id="tabs"></div>
</div>
<iframe id="fr" src="about:blank"></iframe>
<script>
const MAPS = __MAPS__;   // [["name","label","info"],...]
const tabs=document.getElementById('tabs'), fr=document.getElementById('fr');
function show(name){
  fr.src=name+'/web/index.html';
  [...tabs.children].forEach(b=>b.classList.toggle('active',b.dataset.ds===name));
  try{localStorage.setItem('vgp_last',name)}catch(e){}
}
MAPS.forEach(m=>{
  const b=document.createElement('button');
  b.className='tab'; b.dataset.ds=m[0];
  b.innerHTML=m[1]+(m[2]?(' <small>'+m[2]+'</small>'):'');
  b.onclick=()=>show(m[0]);
  tabs.appendChild(b);
});
if(MAPS.length){
  let last=null; try{last=localStorage.getItem('vgp_last')}catch(e){}
  show(MAPS.some(m=>m[0]===last)?last:MAPS[0][0]);
}else{
  document.body.insertAdjacentHTML('beforeend','<div id="empty">暂无已建好的地图</div>');
}
</script></body></html>'''

html = HTML.replace("__MAPS__", json.dumps(maps, ensure_ascii=False))
with open(os.path.join(OUT, "index.html"), "w") as f:
    f.write(html)
print(f"总入口 -> {OUT}/index.html ({len(maps)} 个地图: {[m[0] for m in maps]})")
