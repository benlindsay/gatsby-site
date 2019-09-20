#!/usr/bin/env bash
#
# gh_deploy.sh
#
# Copyright (c) 2019 Ben Lindsay <benjlindsay@gmail.com>

repo="git@github.com:benlindsay/benlindsay.github.io.git"
rm -rf benlindsay.github.io
git clone "$repo"
mv benlindsay.github.io/.git public/
rm -rf benlindsay.github.io
(cd public && git add . && git commit -m "$(date)" && git push origin master)
