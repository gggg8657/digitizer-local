"""Mixed, axisless infographics: printed labels calibrate each series separately."""
import numpy as np
import digitize as D
import layout


def text_boxes(mask, box, multiline=False):
    x0,y0,x1,y1=[int(box[k]) for k in ('x0','y0','x1','y1')]
    m=mask.copy();m[:max(0,y0-25)]=False;m[y1+1:]=False
    m[:,:max(0,x0-30)]=False;m[:,min(m.shape[1],x1+30):]=False
    cs=[c for c in D._components(m) if c[0]>=3 and 5<=c[4]<=30 and c[3]<=50]
    groups=[]
    for c in sorted(cs,key=lambda c:(c[6],c[5])):
        b=[c[5],c[6],c[5]+c[3],c[6]+c[4]]
        for group in groups:
            overlap=min(b[3],group[3])-max(b[1],group[1])
            gap=max(b[0]-group[2],group[0]-b[2],0)
            if overlap>.45*min(b[3]-b[1],group[3]-group[1]) and gap<=max(5,.8*min(b[3]-b[1],group[3]-group[1])):
                group[:]=[min(b[0],group[0]),min(b[1],group[1]),max(b[2],group[2]),max(b[3],group[3])];break
        else:groups.append(b)
    changed=True
    while changed:
        changed=False
        for i,b in enumerate(groups):
            for j,c in enumerate(groups[i+1:],i+1):
                overlap=min(b[3],c[3])-max(b[1],c[1]);gap=max(b[0]-c[2],c[0]-b[2],0)
                if overlap>.45*min(b[3]-b[1],c[3]-c[1]) and gap<=8:
                    groups[i]=[min(b[0],c[0]),min(b[1],c[1]),max(b[2],c[2]),max(b[3],c[3])]
                    groups.pop(j);changed=True;break
            if changed:break
    if multiline:
        merged=[]
        for b in sorted(groups,key=lambda b:b[1]):
            for group in merged:
                overlap=min(b[2],group[2])-max(b[0],group[0])
                if 0<=b[1]-group[3]<=6 and overlap>.7*min(b[2]-b[0],group[2]-group[0]):
                    group[:]=[min(b[0],group[0]),min(b[1],group[1]),max(b[2],group[2]),max(b[3],group[3])];break
            else:merged.append(b)
        groups=merged
    return sorted([b for b in groups if b[2]-b[0]>=7],key=lambda b:(b[0],b[1]))


def auto(a,det,v,read_numbers):
    box=det['box'];sat=a.max(2).astype(float)-a.min(2)
    if not det['axes'].get('synth_y') or len(det['ticks']['y'])>=2:return None
    bars=layout.measure_bars(a,box,pale=True)
    if len(bars)<8:return None
    colors=layout.solid_palette(a,box)
    if not colors:return None
    # The broad, connected saturated trace is independent from the pale bars.
    color=max(colors,key=lambda c:sum(D.mask_for(a,tuple(c.astype(int)),45,box)[0].any(axis=0)))
    rgb=a.astype(float)
    direction=np.array([255,255,255])-color
    fraction=np.clip(((rgb-color)@direction)/np.dot(direction,direction),0,1)
    distance=np.linalg.norm(rgb-(color+fraction[...,None]*direction),axis=2)
    mask=(distance<35)&(fraction<.9)&(sat>20)
    off=(0,0)
    labels,count=D._label_py(mask)
    counts=np.bincount(labels.ravel());counts[0]=0
    if len(counts)<2:return None
    component=int(np.argmax(counts))
    main=labels==component
    trace=D.extract_line(main,off)
    if len(trace)<.65*(box['x1']-box['x0']):return None
    full=np.zeros(a.shape[:2],bool)
    full[off[1]:off[1]+mask.shape[0],off[0]:off[0]+mask.shape[1]]=mask & ~main
    redboxes=text_boxes(full,box)
    readings=read_numbers(a,redboxes) or []
    anchors=[]
    P=np.asarray(trace)
    for b,value in zip(redboxes,readings):
        if value is None:continue
        cx=(b[0]+b[2])/2;cy=(b[1]+b[3])/2
        distances=(P[:,0]-cx)**2+(P[:,1]-cy)**2
        point=P[int(np.argmin(distances))]
        anchors.append((float(value),float(point[1]),b))
    if len(anchors)<2:return None
    low=min(anchors,key=lambda q:q[0]);high=max(anchors,key=lambda q:q[0])
    if abs(high[1]-low[1])<20:return None
    years=v.get('x_ticks') or []
    if len(years)<2:return None
    first,last=float(min(years)),float(max(years))
    # Annual categories only if the observed count proves the regular sequence.
    if abs((last-first+1)-len(bars))>.01:return None
    xcal={'p1':[bars[0]['cx'],box['y1']],'p2':[bars[-1]['cx'],box['y1']],'v1':first,'v2':last,'log':False}
    def ycal(first,second):return {'p1':[box['x0'],first[1]],'p2':[box['x0'],second[1]],'v1':first[0],'v2':second[0],'log':False}
    calib={'x':xcal,'y':ycal(low,high)}
    black=(D.gray(a)<150)&(sat<25)
    blackboxes=text_boxes(black,box,multiline=True)
    blackvalues=read_numbers(a,blackboxes) or []
    # Infographic callout leaders are often 3 px wide at this raster size.
    leaders=[c for c in D._components((D.gray(a)<225)&(sat<20)) if c[3]<=3 and c[4]>=7 and c[4]>=2*c[3]]
    ba=[]
    for b,value in zip(blackboxes,blackvalues):
        if value is None or value<100:continue
        cx=(b[0]+b[2])/2;cy=(b[1]+b[3])/2
        linked=[c for c in leaders if b[0]-10<=c[1]<=b[2]+10 and
                (0<=c[6]-b[3]<=15 or 0<=b[1]-(c[6]+c[4])<=15)]
        if linked:
            leader=min(linked,key=lambda c:abs(c[1]-cx));bar=min(bars,key=lambda q:abs(q['cx']-leader[1]))
        else:bar=min(bars,key=lambda q:(q['cx']-cx)**2+(q['top_px']-cy)**2)
        ba.append((float(value),bar['top_px'],b))
    if len(ba)<2:return None
    bl=min(ba,key=lambda q:q[0]);bh=max(ba,key=lambda q:q[0])
    if abs(bl[1]-bh[1])<15:return None
    calib2={'x':xcal,'y':ycal(bl,bh)}
    D.matrix(calib);D.matrix(calib2)
    lo,hi=sorted((float(calib2['y']['v1']),float(calib2['y']['v2'])))
    if lo<0 or hi<=lo:return None
    bar_color=bars[0]['segs'][0]['sig']['color']
    names=[s.get('name') or '' for s in v.get('legend',[])]
    curve_name=next((s.get('name') for s in v.get('legend',[]) if 'red' in str(s.get('color','')).lower()),None) or (names[0] if names else '선 계열')
    bar_name=next((n for n in names if n!=curve_name),None) or '막대 계열'
    curvepx=[[b['cx'],float(np.interp(b['cx'],P[:,0],P[:,1]))] for b in bars]
    series=[{'name':curve_name,'color':D.rgb2hex(color),'kind':'line','mode':'line','how':'all','px':curvepx,'warn':[]},
            {'name':bar_name,'color':bar_color,'kind':'bar','mode':'points','how':'all','px':[[b['cx'],b['top_px']] for b in bars],'calib':calib2,'axis':'right','warn':[]}]
    det=dict(det,calib=calib,calib2=calib2,series=series,colors=[{'color':s['color'],'count':len(s['px'])} for s in series],legend_boxes=[],fit=None)
    det['warn']=['선과 막대는 서로 다른 척도입니다. 축 눈금이 없어 인쇄된 값과 연결선으로 각각 보정했습니다. 결과는 이미지에서 측정한 근삿값입니다.',
                 '연도 양 끝과 관측한 막대 개수가 일치해 연간 위치로 계산했습니다. 보정 위치를 확인하세요.']
    det['annotation_anchors']={'line':[{'value':x[0],'y':x[1]} for x in anchors],'bar':[{'value':x[0],'y':x[1]} for x in ba]}
    return {'detect':det,'vlm':v,'mode':'xy','bar':None,'panels':[],'need_panel':False,'error':None,
            'message':f'혼합 그래프: 선·막대 {len(series)}계열, 각 {len(bars)}개 연도 위치를 서로 다른 척도로 추출했습니다. 인쇄된 값을 기준으로 보정한 근삿값입니다.'}
