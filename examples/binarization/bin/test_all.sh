#!/bin/bash
  export PYTHONPATH='./src'
MODEL=$1
if [ $MODEL = "otsu" ]; then
   source="country"
   samples="US Canada Mexico..."
else



  export DEVICE='cuda:1'
  export LARGE_DEVICE='cpu'

  python3 -c "import torch;print(sorted(torch.load('$MODEL')['args_history'].items())[-1][-1])"

  RESULT=$(python3 ./bin/train -validationset 'dibco2009' -mode test -device "$DEVICE" -resume_fname "$MODEL" | grep -e 'FS:[0-9\.]*' -o| grep -e '[0-9\.]*' -o)
  RESULT="$RESULT"' & '$(python3 ./bin/train -validationset 'dibco2010' -mode test -device "$DEVICE" -resume_fname "$MODEL" | grep -e 'FS:[0-9\.]*' -o| grep -e '[0-9\.]*' -o)
  RESULT="$RESULT"' & '$(python3 ./bin/train -validationset 'dibco2011' -mode test -device "$DEVICE" -resume_fname "$MODEL" | grep -e 'FS:[0-9\.]*' -o| grep -e '[0-9\.]*' -o)
  RESULT="$RESULT"' & '$(python3 ./bin/train -validationset 'dibco2012' -mode test -device "$DEVICE" -resume_fname "$MODEL" | grep -e 'FS:[0-9\.]*' -o| grep -e '[0-9\.]*' -o)
  RESULT="$RESULT"' & '$(python3 ./bin/train -validationset 'dibco2013' -mode test -device "$DEVICE" -resume_fname "$MODEL" | grep -e 'FS:[0-9\.]*' -o| grep -e '[0-9\.]*' -o)
  RESULT="$RESULT"' & '$(python3 ./bin/train -validationset 'dibco2014' -mode test -device "$DEVICE" -resume_fname "$MODEL" | grep -e 'FS:[0-9\.]*' -o| grep -e '[0-9\.]*' -o)
  RESULT="$RESULT"' & '$(python3 ./bin/train -validationset 'dibco2016' -mode test -device "$DEVICE" -resume_fname "$MODEL" | grep -e 'FS:[0-9\.]*' -o| grep -e '[0-9\.]*' -o)
  RESULT="$RESULT"' & '$(python3 ./bin/train -validationset 'dibco2017' -mode test -device "$DEVICE" -resume_fname "$MODEL" | grep -e 'FS:[0-9\.]*' -o| grep -e '[0-9\.]*' -o)
  RESULT="$RESULT"' & '$(python3 ./bin/train -validationset 'dibco2018' -mode test -device "$DEVICE" -resume_fname "$MODEL" | grep -e 'FS:[0-9\.]*' -o| grep -e '[0-9\.]*' -o)
  RESULT="$RESULT"' & '$(python3 ./bin/train -validationset 'dibco2019' -mode test -device "$LARGE_DEVICE" -resume_fname "$MODEL" | grep -e 'FS:[0-9\.]*' -o| grep -e '[0-9\.]*' -o)
  RESULT="$RESULT"' & '$(python3 ./bin/train -validationset '>dibco2010' -mode test -device "$LARGE_DEVICE" -resume_fname "$MODEL" | grep -e 'FS:[0-9\.]*' -o| grep -e '[0-9\.]*' -o)

  python3 -c "import torch;print(sorted(torch.load('$MODEL')['args_history'].items())[-1][-1])"

  echo "$MODEL:$RESULT"
  echo "$MODEL:$RESULT" >> tmp/eval.log

fi