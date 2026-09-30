import json
import shutil
import subprocess
from pathlib import Path
import pytest
from PIL import Image
from output_media import encode_animation

@pytest.mark.parametrize('fmt', ['mp4', 'webm', 'webp'])
@pytest.mark.parametrize('quality', [1,80,100])
def test_encode(tmp_path, fmt, quality):
    for i in range(1,6):
        Image.new('RGB',(64,64),(i*40,80,120)).save(tmp_path/f'f_{i:05d}_.png')
    dest=tmp_path/f'video.{fmt}'
    encode_animation(str(tmp_path/'f_%05d_.png'),5,24,str(dest),fmt,quality)
    assert dest.stat().st_size>0
    if fmt=='webp':
        with Image.open(dest) as im:
            assert im.n_frames==5 and im.size==(64,64)
            assert im.info['loop']==0
            duration=0
            for i in range(5):
                im.seek(i); im.load(); duration+=im.info['duration']
            assert duration==round(5000/24)
    else:
        data=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-show_streams','-of','json',str(dest)]))['streams'][0]
        assert int(data['nb_read_frames'])==5
        assert data['width']==64 and data['height']==64
        assert data['codec_name']==('h264' if fmt=='mp4' else 'vp9')
        assert abs(float(__import__('fractions').Fraction(data['r_frame_rate']))-24)<.01

@pytest.mark.parametrize('fmt,quality', [('gif',80),('mp4',0),('webp',101)])
def test_invalid(tmp_path,fmt,quality):
    with pytest.raises(ValueError):
        encode_animation('missing_%05d.png',5,24,str(tmp_path/'out'),fmt,quality)
