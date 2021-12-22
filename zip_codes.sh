# Get current directory.
dir=${PWD##*/}

# Zip the current directory.
cd ../ && zip -r COMP551_P3.zip \
  "${dir}" -x \
  "${dir}/.git/*" \
  "${dir}/out/*" \
  "${dir}/dataset/*"\
  "${dir}/.idea/*"\
  "${dir}/.DS_Store"\
  "${dir}/logs/*"
