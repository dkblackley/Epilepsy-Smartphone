#!/bin/bash
# Changes videos to 10fps
FILES="datasets/*"
for f in $FILES
do
  echo "Processing $f file..."
  last_chars=${f: -3}
  file_type="mp4"
  new_name=$"datasets/temp.mp4"
  
  if [[ "$last_chars" == "$file_type" ]]; then
	mv $f $new_name
  	ffmpeg -y -i $new_name -filter:v fps=10 $f
  	rm $new_name
  fi
done
