"""Render reel_linux in N parallel segments, then join and add audio."""
import subprocess, sys, os
from multiprocessing import Process

def seg(i, n, out):
    import importlib; r = importlib.import_module(os.environ.get('REEL_MOD', 'reel_linux'))
    total = int(r.DUR * r.FPS)
    a, b = total * i // n, total * (i + 1) // n
    p = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{r.W}x{r.H}',
                          '-r', str(r.FPS), '-i', '-', '-c:v', 'libx264', '-preset', 'medium', '-crf', '17',
                          '-pix_fmt', 'yuv420p', out], stdin=subprocess.PIPE)
    for k in range(a, b):
        p.stdin.write(r.frame(k / r.FPS).tobytes())
    p.stdin.close(); p.wait()

if __name__ == '__main__':
    out, wav, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
    parts = [f'/tmp/part_{i}.mp4' for i in range(n)]
    ps = [Process(target=seg, args=(i, n, parts[i])) for i in range(n)]
    for p in ps: p.start()
    for p in ps: p.join()
    lst = '/tmp/parts.txt'
    open(lst, 'w').write(''.join(f"file '{p}'\n" for p in parts))
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', lst, '-i', wav,
                    '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k', '-shortest', '-movflags', '+faststart', out], check=True)
    print('done')
