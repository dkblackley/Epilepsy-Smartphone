#!/bin/bash
FILES="datasets/*"
for f in $FILES
do
  echo "Processing $f file..."
  last_chars=${string: -3}
  #file_type = "mp4"
  #mkv_f = ${f%.*}
  #mkv_f = mkv_f += "mkv"
  #echo mkv_f
  new_name=${f%????}"_10.mp4"
  
  if [ "$last_chars" == "$file_type" ];
  then
  
  	#mkvmerge --default-duration 0:12fps --fix-bitstream-timing-information 0 $f -o temp-video.mkv
	#ffmpeg -i temp-video.mkv -c:v copy slow-video.mp4
	#echo $f
	#echo $new_name
  	ffmpeg -y -i $f -filter:v fps=10 $new_name
  fi
done
