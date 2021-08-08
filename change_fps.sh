#!/bin/bash
FILES="datasets/*"
for f in $FILES
do
  echo "Processing $f file..."
  last_chars=${f: -3}
  file_type="mp4"
  #mkv_f = ${f%.*}
  #mkv_f = mkv_f += "mkv"
  #echo mkv_f
  new_name=$"datasets/temp.mp4"
  
  if [[ "$last_chars" == "$file_type" ]]; then
  
  	#mkvmerge --default-duration 0:12fps --fix-bitstream-timing-information 0 $f -o temp-video.mkv
	#ffmpeg -i temp-video.mkv -c:v copy slow-video.mp4
	#echo $f
	#echo $new_name
	
	mv $f $new_name
  	ffmpeg -y -i $new_name -filter:v fps=10 $f
  	rm $new_name
  fi
done
