gource --multi-sampling --output-framerate 60 --seconds-per-day 1.0 --auto-skip-seconds 0.1 ./ -1920x1080 -o - | ffmpeg -y -r 60 -f image2pipe -vcodec ppm -i - -vcodec hevc_nvenc -rc vbr_hq -cq 6 gource.mkv