// Same unclipped linear-sRGB/D65/2-degree contract as difference_visualization.py.
function linearLab(data,i){
  const rgb=[data[i],data[i+1],data[i+2]];
  if(rgb.some(x=>!Number.isFinite(x)||x<0))throw Error('Invalid linear RGB for Delta E00');
  const xyz=[.412453*rgb[0]+.357580*rgb[1]+.180423*rgb[2],.212671*rgb[0]+.715160*rgb[1]+.072169*rgb[2],.019334*rgb[0]+.119193*rgb[1]+.950227*rgb[2]];
  const white=[.95047,1,1.08883];
  const f=xyz.map((x,k)=>{const v=x/white[k];return v>.008856?Math.cbrt(v):7.787*v+16/116;});
  return [116*f[1]-16,500*(f[0]-f[1]),200*(f[1]-f[2])];
}
function ciede2000(a,b){
  const radians=Math.PI/180,cos=x=>Math.cos(x*radians),sin=x=>Math.sin(x*radians);
  const C1=Math.hypot(a[1],a[2]),C2=Math.hypot(b[1],b[2]),C=(C1+C2)/2;
  const G=.5*(1-Math.sqrt(C**7/(C**7+25**7)));
  const ap1=(1+G)*a[1],ap2=(1+G)*b[1],cp1=Math.hypot(ap1,a[2]),cp2=Math.hypot(ap2,b[2]);
  const hue=(aa,bb)=>aa===0&&bb===0?0:(Math.atan2(bb,aa)/radians+360)%360;
  const h1=hue(ap1,a[2]),h2=hue(ap2,b[2]),dL=b[0]-a[0],dC=cp2-cp1;
  let dh=h2-h1;if(cp1*cp2===0)dh=0;else if(dh>180)dh-=360;else if(dh< -180)dh+=360;
  const dH=2*Math.sqrt(cp1*cp2)*sin(dh/2),L=(a[0]+b[0])/2,Cp=(cp1+cp2)/2;
  let H=(h1+h2)/2;if(cp1*cp2===0)H=h1+h2;else if(Math.abs(h1-h2)>180)H+=(h1+h2<360?180:-180);
  const T=1-.17*cos(H-30)+.24*cos(2*H)+.32*cos(3*H+6)-.20*cos(4*H-63);
  const SL=1+.015*(L-50)**2/Math.sqrt(20+(L-50)**2),SC=1+.045*Cp,SH=1+.015*Cp*T;
  const RT=-2*Math.sqrt(Cp**7/(Cp**7+25**7))*sin(60*Math.exp(-(((H-275)/25)**2)));
  const l=dL/SL,c=dC/SC,h=dH/SH;
  return Math.sqrt(Math.max(0,l*l+c*c+h*h+RT*c*h));
}
function numericDifference(a,b,field){
  const n=a.shape[0]*a.shape[1],c=a.shape[2],values=new Float64Array(n);
  if(a.shape.join(',')!==b.shape.join(','))throw Error('Difference shapes mismatch');
  for(let p=0;p<n;p++){
    const i=p*c;
    if(field==='render')values[p]=ciede2000(linearLab(a.data,i),linearLab(b.data,i));
    else if(field==='normal'){
      let dot=0,aa=0,bb=0;for(let k=0;k<3;k++){dot+=a.data[i+k]*b.data[i+k];aa+=a.data[i+k]**2;bb+=b.data[i+k]**2;}
      if(!aa||!bb)throw Error('Zero-length normal');
      values[p]=Math.acos(Math.max(-1,Math.min(1,dot/Math.sqrt(aa*bb))))*180/Math.PI;
    }else if(field==='roughness')values[p]=b.data[i]-a.data[i];
    else{let square=0;for(let k=0;k<c;k++)square+=(b.data[i+k]-a.data[i+k])**2;values[p]=Math.sqrt(square/c);}
    if(!Number.isFinite(values[p]))throw Error('Nonfinite difference');
  }
  return values;
}
function adaptiveDifferenceScale(values){
  const sorted=Array.from(values,Math.abs).sort((a,b)=>a-b),maximum=sorted.at(-1),pos=(sorted.length-1)*.98;
  const lo=Math.floor(pos),hi=Math.ceil(pos),p98=sorted[lo]+(sorted[hi]-sorted[lo])*(pos-lo);
  return {upper:p98>0&&p98<maximum/2?p98:maximum,maximum,policy:p98>0&&p98<maximum/2?'P98 absolute':'full extrema'};
}
function differenceColor(t,signed){
  const stops=signed?[[59,76,192],[221,221,221],[180,4,38]]:[[68,1,84],[59,82,139],[33,145,140],[94,201,98],[253,231,37]];
  const x=Math.max(0,Math.min(1,t))*(stops.length-1),i=Math.min(stops.length-2,Math.floor(x)),f=x-i;
  return stops[i].map((v,k)=>Math.round(v+(stops[i+1][k]-v)*f));
}
