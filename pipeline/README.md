# Daily Reel Pipeline v2 (cinematic engine + George voice)

One run = one finished reel for **the day after the run** (if that day has no reel yet), posted to YouTube Shorts, TikTok and Instagram Reels through Metricool, plus one row in the pipeline log. Never more than one reel per day.

- Metricool brand **dailycodingnews**, blogId `7479861`, timezone `Asia/Jerusalem`. Instagram is @dailycodenews.
- Pipeline log: artifact https://claude.ai/artifact/KF4L5Cyt3Mu7Mjp5JH4rRM, collection `posts` (ArtifactData).
- Quality bar = the reels made on Oct 22-24 (`videos/reel_2026-10-23_php_v2.mp4`, `videos/reel_2026-10-24_neural.mp4`): dramatic hook in the first 2 seconds, beginner friendly but with a few real technical ideas shown visually, cinematic and realistic (depth, glow, fog, light shafts, handheld camera, floating dust are built into the engine), black + light blue only, 45-60 s.

## 0. Setup (fresh session)
```
cd /home/claude && git clone https://github.com/asaf-beshkin/dailycodingnews-media.git   # public, read works
mkdir -p /home/claude/work && cp dailycodingnews-media/pipeline/*.py /home/claude/work/ && cp -r dailycodingnews-media/pipeline/examples /home/claude/work/
cd /home/claude/work
for f in v1.13.8/sherpa-onnx-v1.13.8-linux-x64-static.tar.bz2 tts-models/kokoro-multi-lang-v1_0.tar.bz2 asr-models/sherpa-onnx-zipformer-en-2023-06-26.tar.bz2; do
  curl -sfL -O "https://github.com/k2-fsa/sherpa-onnx/releases/download/$f" && tar xjf "$(basename $f)"; done
```
`voice2.py`, `make_timing.py` and `cut_takes.py` expect the three unpacked folders in the same directory. numpy, scipy, Pillow and ffmpeg are installed; pip does not work, don't route around network blocks.

## 1. Which day, which topic
- `getScheduledPosts` (brandId 7479861, Asia/Jerusalem) for tomorrow. If tomorrow already has a post, stop and log nothing (the reel exists). Target = tomorrow.
- Read the `posts` log; never repeat a topic that was posted or scheduled.
- Topic: an everyday-tech or how-it-works topic people search for, with a natural visual story (examples: TCP vs UDP, how HTTPS keeps you safe, how GPS finds you, how compression shrinks files, how a CPU runs code, how DNS finds a website, how passwords are stored). Search the web for what is trending or newly in the news in tech, and pick the strongest. Check every fact against two pages you opened.
- Security news is fine at the level of "what happened and what it means for you"; never explain how an attack works.

## 2. Script (about 125-140 words, 11-13 caption lines)
- **Hook (line 0)**: a surprising concrete fact, a question or a stakes line, under 2.5 s to the payoff word. Examples that worked: "Your brain has about 86 billion neurons. AI copied the idea... using nothing but numbers." / "This week, two hidden flaws were found in the engine behind seven out of ten websites."
- Then: name the thing simply → how it works in 2-4 concrete steps (show the real mechanism with small real numbers or labels, a bit technical, every term explained in the same sentence) → a twist or funny beat → the payoff → a one-line closer that calls back to the hook.
- Short sentences, everyday words. Digits are fine on screen; put their spoken form in `SPOKEN`.

## 3. Voice: George (vidIQ) first, free fallback
1. `vidiq_balance`. If at least 15 credits: `vidiq_voiceover_generate` with voiceId `JBFqnCBsd6RMkjVDRZzb` (George, British male) and the whole spoken script as one text (use "..." for dramatic pauses). Poll `vidiq_job_poll`, then `curl -sfL -o narration.mp3 "<audioUrl>"`.
2. If credits are below 15, or the download is blocked: free Kokoro voice in one take: `python3 -c "import voice2 as v; v.tts(open('spoken.txt').read(), 'narration.wav', sid=26); v.polish('narration.wav')"`.
3. If the narration is over 60 s, speed it up with `ffmpeg -i narration.mp3 -af atempo=1.06 narration.wav` (max 1.08).
4. Word timing: write `lines.json` like `examples/lines_neural_network.json` (caption lines + `spoken` map), then `python3 make_timing.py narration.wav lines.json timing.js`. Check the printout (recognizer spelling quirks are fine; missing words are not).
5. Cut one take per scene: `python3 cut_takes.py narration.wav timing.js '[[0],[1],[2],[3,4],[5,6],[7,8,9],[10],[11]]' takes` (groups = the scene letters of your LINES, in order).

## 4. Build the film
1. `cp reel_engine.py reel_today.py`. Replace: `LINES` (scene letter + caption text, same order as the narration), `SPOKEN`, `GAPS` (extra pause before a line, seconds), `BUGTXT` (small corner label, e.g. "HOW TECH WORKS" or "TECH NEWS"), and the whole block between `# ===== EPISODE START` and `# ===== EPISODE END` with new shots for today's topic. Keep the names `bug`, `scA..scH`, `SCENEFN`, `HITS`. Read `examples/episode_neural_network.py` and `examples/episode_php_news.py` first: they show the patterns (3D projection with depth blur, `glow`, `rimlit` silhouettes, `corridor` server hall, `globe_scene`, phone in hand, metal objects, `pill`/`tsp`/`put`, `wt(line, word)` for timing every hit to the spoken word).
2. Every scene: something moves all the time, one main idea, labels of 4 words or fewer, nothing important in the top 200 px or below y=1250 (captions live there), a camera move (slow push-in or orbit), and at least one realistic touch (depth, light, reflection, texture).
3. `VO_DIR=takes python3 -c "import reel_today as r; r.print_script()"`: total 45-62 s.
4. Review: render 12-16 frames across every scene into a contact sheet and LOOK at it. Fix overlaps, empty moments, things cut off at the edges, unreadable text. Repeat until clean (at least two review rounds).
5. `VO_DIR=takes python3 reel_today.py audio day.wav`, then `VO_DIR=takes REEL_MOD=reel_today python3 render_parallel.py raw.mp4 day.wav 2` (about 15 min, run in background), then `ffmpeg -i raw.mp4 -c:v libx264 -preset slow -crf 21 -maxrate 9M -bufsize 18M -c:a copy -movflags +faststart reel_YYYY-MM-DD.mp4` (must be under 100 MB).
6. Quality gate: pull 6 frames from the final MP4 and check them. Captions must match the voice.

## 5. Captions for each platform
- YouTube: `title` ≤ 70 chars, a curiosity hook, ends with `#Shorts`; 6-9 search `tags`; category SCIENCE_TECHNOLOGY, madeForKids false, type short, privacy public, isAiGeneratedContent false.
- Shared `text` (TikTok/Instagram/YouTube description): hook + one line of value + 4-5 hashtags. tiktokData: short `title`, PUBLIC_TO_EVERYONE, isAigc true. instagramData: REEL, showReelOnFeed true, isAiGenerated true.

## 6. Upload to the media repo
`add_repo` (owner asaf-beshkin, repo dailycodingnews-media, access push), then in the clone: copy the MP4 to `videos/reel_YYYY-MM-DD.mp4`, commit with `-c user.name="dailycodingnews pipeline" -c user.email="asafbekhin@gmail.com"`, `git push origin main`. Link for Metricool: `https://raw.githubusercontent.com/asaf-beshkin/dailycodingnews-media/main/videos/reel_YYYY-MM-DD.mp4`.
If add_repo or the push fails: don't try other routes. Log `blocked` with the exact error, send the MP4 and captions to the owner, stop.

## 7. Schedule
`getBestTimeToPostByNetwork` (tiktok and instagram) for the target day; strongest hour at least 20 h after the previous post (fallback 10:00 Asia/Jerusalem); use a minute like :03 rather than :00. One `createScheduledPost` with providers youtube + tiktok + instagram, autoPublish true.

## 8. Log and report
ArtifactData `set` in `posts`, doc id `YYYY-MM-DD`: {date, topic, hook, status (scheduled | blocked | failed), scheduledFor, platforms, youtubeTitle, youtubeTags, tiktokTitle, tiktokCaption, plannerUrl, videoLink, notes (sources, voice used, length)}. Send the MP4 to the owner (SendUserFile, status proactive) with one line: topic, post time, status.
