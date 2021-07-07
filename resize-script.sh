#!/bin/bash

#simple script for resizing images in all class directories
#also reformats everything from whatever to png

if [ `ls ./data/27_wm/*/*.jpg 2> /dev/null | wc -l ` -gt 0 ]; then
  echo hi
  for file in ./data/27_wm/*/*.jpg; do
    convert "$file" -resize 28x28\! "${file%.*}.png"
    file "$file" #uncomment for testing
    rm "$file"
  done
fi

if [ `ls ./data/27_wm/*/*.png 2> /dev/null | wc -l ` -gt 0 ]; then
  echo hi
  for file in ./data/27_wm/*/*.png; do
    convert "$file" -resize 28x28\! "${file%.*}.png"
    file "$file" #uncomment for testing
  done
fi

if [ `ls ./data/27_wm/*/*.jpg 2> /dev/null | wc -l ` -gt 0 ]; then
  echo hi
  for file in ./data/27_wm/*/*.jpg; do
    convert "$file" -resize 28x28\! "${file%.*}.png"
    file "$file" #uncomment for testing
    rm "$file"
  done
fi

if [ `ls ./data/27_wm/*/*.png 2> /dev/null | wc -l ` -gt 0 ]; then
  echo hi
  for file in ./data/27_wm/*/*.png; do
    convert "$file" -resize 28x28\! "${file%.*}.png"
    file "$file" #uncomment for testing
  done
fi
