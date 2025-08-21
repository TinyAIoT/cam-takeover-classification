#!/bin/sh
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$SCRIPT_DIR"
# download data from sciebo
wget -nc -O classified.zip "https://uni-muenster.sciebo.de/s/cQCSXCWbbow8wA3/download?path=%2F&files=classified.zip"
# unpack data
mkdir -p classified
mv classified.zip classified/
unzip classified/classified.zip -d classified/
rm classified/classified.zip