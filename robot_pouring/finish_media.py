"""Encode verified frames with truthful task telemetry and build a local gallery."""
import argparse
from functools import lru_cache
import hashlib
import html
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont

TITLES={"basin":"Basin Transfer", "precision":"Precision Pour", "flower":"Gentle Watering"}
from physstudio.paths import data_path

DEFAULT_FONT_DIR=data_path("robot_pouring", "fonts")


@lru_cache(maxsize=16)
def font(font_dir, size, bold=False):
    """Use the same Google Sans files as the accepted PhysStudio overview."""
    return ImageFont.truetype(str(Path(font_dir)/f'GoogleSans-{700 if bold else 400}.ttf'),size)


@lru_cache(maxsize=3)
def title_overlay(title, font_dir):
    overlay=Image.new('RGBA',(1280,720));draw=ImageDraw.Draw(overlay)
    # Match the overview's translucent gradients, proportional margins and type.
    for y in range(180):
        alpha=round(190*max(0,1-max(0,y-32)/148)**1.4)
        draw.line((0,y,1280,y),fill=(15,14,13,alpha))
    for y in range(604,720):
        draw.line((0,y,1280,y),fill=(16,15,14,round(220*((y-604)/115)**.8)))
    draw.text((43,24),title,anchor='lt',font=font(font_dir,38,True),fill='#f6f3ec')
    draw.text((44,75),'Robot Pouring · Reference controller',anchor='lt',
              font=font(font_dir,18),fill='#e1ddd3')
    draw.text((1236,30),'PhysStudio',anchor='rt',font=font(font_dir,18,True),fill='#e1ddd3')
    draw.line((44,108,76,108),fill='#d8c7aa',width=2)
    for x,label in [(44,'Captured'),(220,'Spilled'),(396,'Goal')]:
        draw.text((x,649),label,anchor='lt',font=font(font_dir,15),fill='#e1ddd3')
    return overlay


def compose_frame(source, title, captured, spilled, goal_ml, elapsed, font_dir):
    overlay=title_overlay(title,str(font_dir)).copy();draw=ImageDraw.Draw(overlay)
    for x,value in [(44,f'{captured*100:.1f}%'),(220,f'{spilled*100:.1f}%'),
                    (396,f'{goal_ml:.0f} mL')]:
        draw.text((x,674),value,anchor='lt',font=font(str(font_dir),23),fill='#f6f3ec')
    draw.text((1236,678),f'{elapsed:.2f} s',anchor='rt',font=font(str(font_dir),18),fill='#e1ddd3')
    if overlay.size != source.size:
        overlay=overlay.resize(source.size,Image.Resampling.LANCZOS)
    return Image.alpha_composite(source.convert('RGBA'),overlay).convert('RGB')


def finish(root, output, font_dir, frames=None):
    config=json.loads((root/'config.json').read_text());report=json.loads((root/'report.json').read_text())
    audit=json.loads((root/'audit.json').read_text());assert audit['passed']
    frames=frames or root/'frames';expected=list(range(report['frames']))
    assert sorted(int(p.stem) for p in frames.glob('*.png'))==expected
    stage=output/(config['task']+'_frames');stage.mkdir()
    title=TITLES[config['task']]
    for i in expected:
        with Image.open(frames/f'{i:05d}.png') as source:
            source.load();im=source.convert('RGBA')
        with np.load(root/'states'/f'{i:05d}.npz') as d:
            mass=d['mass_fractions'];captured=float(mass[1]+mass[4]);spill=float(mass[3]);time=float(d['time'])
        im=compose_frame(im,title,captured,spill,audit['requested_ml'],time,font_dir)
        im.save(stage/f'{i:05d}.png')
    video=output/(config['task']+'.mp4')
    fps=1/(config['dt']*config['substeps'])
    subprocess.run(['ffmpeg','-v','error','-y','-framerate',str(fps),'-i',str(stage/'%05d.png'),
                    '-c:v','libx264','-threads','6','-preset','slow','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],check=True)
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0',
                    '-show_entries','stream=width,height,r_frame_rate,nb_read_frames:format=duration','-of','json',str(video)]))
    assert int(probe['streams'][0]['nb_read_frames'])==report['frames']
    subprocess.run(['ffmpeg','-v','error','-i',str(video),'-f','null','-'],check=True)
    peak=max(report['records'],key=lambda row:row['mass'][2])['frame']
    with Image.open(stage/f'{peak:05d}.png') as im:im.save(output/(config['task']+'.jpg'),quality=94)
    thumbnails=[]
    for i in [0,peak,report['frames']-1]:
        with Image.open(stage/f'{i:05d}.png') as im:thumbnails.append(im.resize((640,360)))
    sheet=Image.new('RGB',(1920,360))
    for j,im in enumerate(thumbnails):sheet.paste(im,(640*j,0))
    sheet.save(output/(config['task']+'_storyboard.jpg'),quality=94)
    return dict(task=config['task'],title=title,path=str(video.resolve()),probe=probe,
                sha256=hashlib.sha256(video.read_bytes()).hexdigest(),audit=audit,controller='reference',full_decode=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--font-dir',type=Path,default=DEFAULT_FONT_DIR,
                   help='Directory containing the overview GoogleSans-400.ttf and GoogleSans-700.ttf')
    p.add_argument('--frames-dir',type=Path,help='Rendered PNG directory for a single input recording')
    args=p.parse_args()
    if args.frames_dir and len(args.inputs)!=1: p.error('--frames-dir requires one input recording')
    args.output.mkdir(parents=True,exist_ok=False)
    font_dir=args.output/'fonts';font_dir.mkdir()
    fonts=[]
    for weight in [400,700]:
        source=args.font_dir/f'GoogleSans-{weight}.ttf'
        target=font_dir/source.name;shutil.copy2(source,target)
        family,style=ImageFont.truetype(str(target),18).getname()
        fonts.append(dict(file=str(target.relative_to(args.output)),family=family,style=style,
                          source=str(source.resolve()),sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
    license_source=args.font_dir/'OFL.txt'
    if not license_source.exists():
        license_source=Path(__file__).with_name('references')/'GoogleSans-OFL.txt'
    shutil.copy2(license_source,font_dir/'OFL.txt')
    typography=dict(reference='PhysStudio overview typography: Google Sans 700/400',
                    fonts=fonts,title_size=38,body_size=18,metric_label_size=15,metric_value_size=23,
                    treatment='Overview typography with translucent gradients and proportional margins',
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.output/'typography.json').write_text(json.dumps(typography,indent=2)+'\n')
    records=[finish(root,args.output,font_dir,args.frames_dir) for root in args.inputs]
    (args.output/'manifest.json').write_text(json.dumps(records,indent=2)+'\n')
    cards=[]
    for r in records:
        task=r['task'];a=r['audit']
        cards.append(f'<section><h2>{html.escape(r["title"])}</h2><video controls loop playsinline preload="metadata" poster="{task}.jpg" src="{task}.mp4"></video><p>Captured {a["final_captured_fraction"]*100:.1f}% · Spilled {a["final_spilled_fraction"]*100:.1f}% · {a["particles"]:,} simulated water particles</p></section>')
    page='''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PhysStudio · Robot Pouring</title><style>
@font-face{font-family:Google Sans;src:url(fonts/GoogleSans-400.ttf);font-weight:400}
@font-face{font-family:Google Sans;src:url(fonts/GoogleSans-700.ttf);font-weight:700}
body{margin:0;background:#0c131c;color:#f6f3ec;font:17px 'Google Sans',sans-serif}
main{max-width:1200px;margin:auto;padding:48px 24px}h1{font-size:42px;margin-bottom:12px}
p{color:#c5c5c0;line-height:1.6}section{margin:42px 0}
video{width:100%;border-radius:14px;background:#000}h2{font-size:26px}
</style><main><h1>Robot Pouring</h1>
<p>Franka Panda · MuJoCo Warp + Warp particle water · GPU reinforcement-learning environment</p>
<p>These videos use a feedback reference controller acting through the same joint-action interface available to an RL policy.</p>
'''+''.join(cards)+'</main></html>'
    (args.output/'index.html').write_text(page)
    print(json.dumps([dict(task=r['task'],path=r['path'],sha256=r['sha256']) for r in records],indent=2))


if __name__=='__main__':main()
