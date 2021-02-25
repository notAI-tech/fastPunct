rm fastDeploy.py
wget https://raw.githubusercontent.com/notAI-tech/fastDeploy/master/cli/fastDeploy.py

python3 fastDeploy.py --build temp --source_dir fastDeploy_recipes --verbose --base pyt_1_5_cpu --port 1238 --extra_config '{"MODEL_NAME": "english"}'
docker commit temp notaitech/fastpunct:english
docker push notaitech/fastpunct:english

rm fastDeploy.py
