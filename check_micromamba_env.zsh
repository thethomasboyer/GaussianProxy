#!/usr/bin/env zsh

if [[ ! -f environment.yaml ]]; then
   echo "environment.yaml not found!"
   exit 1
fi

# Export the environment and remove trailing newlines
micromamba env export | sed -e :a -e '/^\n*$/{$d;N;};/\n$/ba' >tmp_env.yaml

diff -q environment.yaml tmp_env.yaml >/dev/null

# If environments match, exit successfully with no output
[[ $? -eq 0 ]] && rm tmp_env.yaml && exit 0

# If there's a mismatch, print the diff
echo "Activated micromamba environment does not match environment.yaml"
echo "environment.yaml vs currently activated environment:"
diff --color=always -y --suppress-common-lines environment.yaml tmp_env.yaml

# Remove the temporary file
rm tmp_env.yaml

exit 1
