const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const html=fs.readFileSync('live_training_plot.html','utf8');
const script=html.match(/<script>([\s\S]*?)<\/script>/)[1];
const elements=new Map();let selectedScale='linear',selectedCategory='overall';
const ctx=new Proxy({}, {get:(obj,key)=>obj[key]||(()=>{})});
function element(id){if(!elements.has(id))elements.set(id,{value:id==='#scale'?selectedScale:selectedCategory,clientWidth:900,getContext:()=>ctx,addEventListener(){},replaceChildren(){},append(){},style:{}});return elements.get(id);}
const snapshot={train:{'map:normal_loss':[[1,1,.9],[2,.2,.55]],'map:roughness_loss':[[1,0,0],[2,.01,.005]]},val:{'map:map_normal':[[0,.8],[2,.3]],'map:Fabric:map_normal':[[0,1.2]]},status:{status:'training'},plateau:{},latest:{total_step:2,lr:.0005},note:''};
const sandbox={document:{querySelector:element,createElement:()=>element(Symbol()),createTextNode:x=>x},window:{addEventListener(){}},devicePixelRatio:1,setInterval(){},fetch:async()=>({ok:true,json:async()=>snapshot}),console};
vm.createContext(sandbox);vm.runInContext(script,sandbox);
const linear=vm.runInContext("sharedBounds([...collect("+JSON.stringify(snapshot.train)+",true),...collect("+JSON.stringify(snapshot.val)+",false)],'linear')",sandbox);
assert.equal(linear.ymin,0);assert.equal(linear.ymax,1.2*1.05);assert.equal(linear.xmax,2);
const log=vm.runInContext("sharedBounds([...collect("+JSON.stringify(snapshot.train)+",true),...collect("+JSON.stringify(snapshot.val)+",false)],'log')",sandbox);
assert.equal(log.ymin,.001);assert.equal(log.ymax,10);
assert.equal(log.xmin,linear.xmin);assert.equal(log.xmax,linear.xmax);
assert.match(html,/const X=x=>left\+\(x-bounds.xmin\)\/\(bounds.xmax-bounds.xmin\)/);
sandbox.bounds=log;assert.equal(vm.runInContext('projectY(0,bounds)',sandbox),null);
assert.equal(vm.runInContext('projectY(bounds.ymin,bounds)',sandbox),0);
assert.equal(vm.runInContext('projectY(bounds.ymax,bounds)',sandbox),1);
vm.runInContext('latest='+JSON.stringify(snapshot)+'; render()',sandbox);
const range=element('#range').textContent;
element('#category').value='Leather';vm.runInContext('render()',sandbox);assert.equal(element('#range').textContent,range);
element('#scale').value='log';vm.runInContext('render()',sandbox);assert.match(element('#range').textContent,/log/);
assert.match(html,/id="train"/);assert.match(html,/id="val"/);
console.log('shared linear/log ranges, zero handling, category invariance and canvas rendering: PASS');

assert.equal(vm.runInContext("JSON.stringify(yTicks(bounds).map(t=>t.label))",sandbox),JSON.stringify(['10⁻³','10⁻²','10⁻¹','10⁰','10¹']));

assert.equal(vm.runInContext('metricValue(null)',sandbox),'—');
assert.equal(vm.runInContext('metricValue(0)',sandbox),'0');
for(const id of ['status-table','train-table','val-table'])assert.ok(html.includes('id="'+id+'"'));

assert.ok(!html.includes('Normal angle median'));
assert.ok(!html.includes('Normal angle P95'));
assert.ok(html.includes('Normal angle — 재질별 평균'));

sandbox.trainerPlateau={source:'trainer',status:'floor_wait',metric_label:'NDRS sum',streak:7,target:20,meaningful_best:.120571};
const trainerText=vm.runInContext("plateauText('map',trainerPlateau)",sandbox);
assert.match(trainerText,/NDRS sum/);assert.match(trainerText,/종료 대기 7\/20/);
assert.ok(!trainerText.includes('/10'));assert.ok(!trainerText.includes('종료 조건 충족'));
assert.equal(vm.runInContext("plateauLabel({...trainerPlateau,status:'not_started'})",sandbox),'미시작');
assert.equal(vm.runInContext("plateauLabel({...trainerPlateau,status:'plateau'})",sandbox),'종료 조건 충족');
assert.equal(vm.runInContext("plateauCount({...trainerPlateau,target:8})",sandbox),'7/8');
assert.match(vm.runInContext("plateauText('map',{status:'plateau',streak:24})",sandbox),/기존 진단 기준/);
console.log('trainer metric, recorded floor counter/target, phase state and legacy fallback: PASS');
