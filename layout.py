"""Plot bounds from repeated grid lines / bar baselines, including axisless charts.
No model coordinates: spreadsheet borders and infographic frames are excluded.
"""
import numpy as np
import digitize as D


def grid_boxes(a):
    H,W=a.shape[:2]
    groups=[]
    for line in D._line_groups(D.gray(a)<235,0):
        if line[2]<.35*W or line[3]<.02*W or line[4]>.99*W: continue
        for group in groups:
            if abs(line[3]-group[0][3])<=4 and abs(line[4]-group[0][4])<=4:
                group.append(line);break
        else:groups.append([line])
    boxes=[]
    for group in groups:
        if len(group)<3:continue
        ys=sorted(x[0] for x in group);diff=np.diff(ys)
        if max(diff)/min(diff)>1.25:continue
        boxes.append([float(group[0][3]),float(ys[0]),float(group[0][4]+1),float(ys[-1])])
    return boxes


def bar_groups(a):
    H,W=a.shape[:2];sat=a.max(2).astype(float)-a.min(2);gray=D.gray(a)
    components=[]
    # Separating pale bars from saturated lines avoids joining a mixed plot into one component.
    for mask in ((sat>18)&(sat<80)&(gray<245),(sat>=80)&(gray<245)):
        components += [c for c in D._components(mask) if 3<=c[3]<=.07*W and c[4]>=1 and c[0]/(c[3]*c[4])>.60]
    groups=[]
    for c in sorted(components,key=lambda c:c[6]+c[4]):
        bottom=c[6]+c[4]
        for group in groups:
            if abs(bottom-np.median([g[6]+g[4] for g in group]))<=3:group.append(c);break
        else:groups.append([c])
    out=[]
    for group in groups:
        left=min(c[5] for c in group);right=max(c[5]+c[3] for c in group)
        if len(group)<8 or right-left<.45*W or sum(c[4]>=max(16,.02*H) for c in group)<5:continue
        base=float(np.median([c[6]+c[4] for c in group]))
        # Join the same bar across thin black segment boundaries / an overlaid line.
        ink=sat>18
        tops=[]
        for c in group:
            center=int(round(c[1]));last=int(c[6]+c[4]-1);blank=0;first=last
            for yy in range(last,-1,-1):
                if ink[yy,max(0,center-1):center+2].any():first=yy;blank=0
                else:
                    blank+=1
                    if blank>4:break
            tops.append(first)
        top=float(min(tops));gap=float(np.median([c[3] for c in group]))/2
        # A real black baseline and dotted neutral grid give exact panel bounds.
        hs=D._line_groups(D.gray(a)<160,0)
        near=[h for h in hs if abs(h[0]-base)<=6 and h[3]<=left and h[4]>=right-3]
        if near:
            line=min(near,key=lambda h:abs(h[0]-base));base=line[0];left=line[3];right=line[4]+1
        else:left-=gap;right+=gap
        neutral=(D.gray(a)<190)&(sat<20)
        grid=[]
        for y in range(max(0,int(top)-int(.3*H)),int(base)):
            rs=D.merge(D.runs(neutral[y,max(0,int(left)):min(W,int(right))]),max(5,int(.04*(right-left))))
            if any(e-s>.85*(right-left) for s,e in rs):grid.append(y+.5)
        if grid:
            bands=D.runs(np.isin(np.arange(H),[int(y) for y in grid]))
            levels=[(s+e)/2+.5 for s,e in bands if e-s<=3]
            if len(levels)>=2:
                chain=[base];step=None
                for level in sorted(levels,reverse=True):
                    gap=chain[-1]-level
                    if gap<4:continue
                    if step is not None and not .65*step<=gap<=1.4*step:break
                    chain.append(level);step=gap if step is None else step
                if len(chain)>=3:top=min(top,min(chain))
        # Trace may rise above the tallest pale bar (mixed infographic).
        colored=(sat>80)&(D.gray(a)<245)
        line_parts=[c for c in D._components(colored) if c[3]>.65*(right-left) and c[5]>=left-10 and c[5]+c[3]<=right+10 and c[6]+c[4]<base]
        if line_parts:top=min(top,min(c[6] for c in line_parts))
        out.append(dict(box=[max(0,float(left)),max(0,top),min(W-1,float(right)),base],bars=sorted(group,key=lambda c:c[1])))
    return out


def plot_boxes(a):
    boxes=grid_boxes(a)+[g['box'] for g in bar_groups(a)]
    boxes.sort(key=lambda b:-(b[2]-b[0])*(b[3]-b[1]))
    unique=[]
    for b in boxes:
        if b[3]-b[1]<25:continue
        if any(o[0]-8<=b[0] and o[1]-8<=b[1] and b[2]<=o[2]+8 and b[3]<=o[3]+8 for o in unique):continue
        unique.append(b)
    return sorted(unique,key=lambda b:(b[1],b[0]))


def panels(a):
    H,W=a.shape[:2];out=[]
    for b in plot_boxes(a):
        x0,y0,x1,y1=b
        out.append({'box':b,'crop':[max(0,x0-.16*(x1-x0)),max(0,y0-35),min(W,x1+15),min(H,y1+50)],'kind':'plot','score':1.0})
    return out


def measure_bars(a, box, pale=False):
    """Center strips of solid colored bars; one-pixel stacks remain measurable.
    White gaps over four pixels stop a bar, excluding legends floating above it.
    """
    x0,y0,x1,base=[float(box[k]) for k in ('x0','y0','x1','y1')]
    sat=a.max(2).astype(float)-a.min(2)
    mask=(sat>18)&(D.gray(a)<245)&(a.max(2)>60)
    if pale:
        mask &= sat<80
        samples=a[max(0,int(base)-5):int(base),max(0,int(x0)):int(x1)]
        samplemask=mask[max(0,int(base)-5):int(base),max(0,int(x0)):int(x1)]
        if samplemask.any():
            color=np.median(samples[samplemask],axis=0)
            mask &= np.linalg.norm(a.astype(float)-color,axis=2)<45
        palette=[color] if samplemask.any() else []
    else:
        palette=solid_palette(a,box)
    lo,hi=max(0,int(x0)+2),min(a.shape[1],int(x1)-1)
    band=mask[max(0,int(base)-5):int(base),lo:hi]
    runs=D.runs(band.any(axis=0))
    bars=[]
    for l,r in runs:
        if not 3<=r-l+1<=.07*a.shape[1]:continue
        l+=lo;r+=lo;cx=(l+r+1)/2
        center=int(cx);strip=slice(max(l,center-1),min(r+1,center+2))
        rowmask=mask[:,strip].any(axis=1)
        last=int(base)-1;blank=0;top=last
        colors=[]
        for yy in range(last,max(0,int(y0))-1,-1):
            if rowmask[yy]:
                top=yy;blank=0
                px=a[yy,strip][mask[yy,strip]]
                color=px.astype(float).mean(axis=0)
                if palette:color=min(palette,key=lambda p:np.linalg.norm(color-p))
                colors.append((yy,color))
            else:
                # Colored annotation / line drawn ON a pale bar is an occlusion,
                # not the top of that bar. Do not bridge a white legend gap.
                if pale and D.gray(a[yy,strip]).min()<240:blank=0
                else:blank+=1
                if blank>4:break
        if not colors:continue
        # Dominant palette rather than hue extrapolation, including thin red caps.
        segments=[];group=[]
        for yy,color in colors:
            if group and np.linalg.norm(color-np.median([c for _,c in group],axis=0))>65:
                segments.append(group);group=[]
            group.append((yy,color))
        if group:segments.append(group)
        cleaned=[];seen=[]
        for segment in segments:
            color=segment[0][1]
            if cleaned and len(segment)<=2 and any(np.linalg.norm(color-c)<20 for c in seen):
                cleaned[-1].extend((y,cleaned[-1][0][1]) for y,_ in segment)
            else:cleaned.append(segment);seen.append(color)
        segments=cleaned
        segs=[];bottom=base
        for group in segments:
            end=min(y for y,_ in group)
            col=np.median([c for _,c in group],axis=0)
            segs.append({'y0':bottom,'y1':float(end),'sig':{'kind':'solid','color':D.rgb2hex(col),'ink':1.0}})
            bottom=float(end)
        bars.append({'l':float(l),'r':float(r+1),'cx':cx,'top_outer':float(top),'top_px':float(top),
                     'segs':segs,'err':{},'open':False})
    return bars


def solid_palette(a, box):
    x0,y0,x1,y1=[int(box[k]) for k in ('x0','y0','x1','y1')]
    rgb=a[max(0,y0):y1,max(0,x0):x1].reshape(-1,3).astype(float)
    rgb=rgb[(rgb.max(1)-rgb.min(1)>55)&(rgb.max(1)>80)]
    if not len(rgb):return []
    q=(rgb//16).astype(int);keys=q[:,0]*256+q[:,1]*16+q[:,2]
    unique,inv,count=np.unique(keys,return_inverse=True,return_counts=True)
    palette=[]
    for i in np.argsort(-count):
        if count[i]<8:break
        color=rgb[inv==i].mean(0)
        if any(np.linalg.norm(color-p)<70 for p in palette):continue
        white=np.array([255,255,255])
        blend=False
        for p in palette:
            direction=white-p;fraction=np.clip(np.dot(color-p,direction)/np.dot(direction,direction),0,1)
            if fraction>.1 and np.linalg.norm(p+fraction*direction-color)<40:blend=True;break
        if blend:continue
        for i,first in enumerate(palette):
            for second in palette[i+1:]:
                direction=second-first;fraction=np.clip(np.dot(color-first,direction)/max(1,np.dot(direction,direction)),0,1)
                if .1<fraction<.9 and np.linalg.norm(first+fraction*direction-color)<35:blend=True
        if blend:continue
        palette.append(color)
        if len(palette)>=8:break
    return palette
