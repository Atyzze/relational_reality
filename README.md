download main.py & requirements.txt and open console in your download directory (or git clone ofc)

#create local virtual environment for your python snek to run in

python -m venv venv

#ask your LLM/IT-person of preference to match your local OS

source venv/bin/activate 

pip install -r requirements.txt

python main.py 

the default is set to N100 and takes about a minute to calculcate on a decade old budget laptop

python main.py -N 12800

results below, without setting step count, it defaults to N * 3000 as step count, to let it fully crystalize.

<img width="4694" height="2367" alt="N=12800_S=38400000_m" src="https://github.com/user-attachments/assets/704210a5-c169-4097-8a36-72a29d54a849" />

<img width="5330" height="2954" alt="N=12800_S=38400000_a" src="https://github.com/user-attachments/assets/183313f3-30d4-4952-9b2b-6c9e0b2d843a" />

<img width="2970" height="1191" alt="N=12800_S=38400000_d" src="https://github.com/user-attachments/assets/a4c79d25-bba8-483d-a4a3-a85303488376" />

<img width="2956" height="1775" alt="N=12800_S=38400000_g" src="https://github.com/user-attachments/assets/291ae796-5c6d-4543-b968-43fbef4e0da0" />

<img width="3177" height="1948" alt="triangles_scaling" src="https://github.com/user-attachments/assets/a9e532b7-85f3-4e97-9ba3-1b272f564de1" />

<img width="3087" height="1948" alt="k_mean_scaling" src="https://github.com/user-attachments/assets/60b5b64b-2e7d-4b1d-9837-fa48379fc1e3" />

<img width="2972" height="1948" alt="hausdorff_dim_scaling" src="https://github.com/user-attachments/assets/10841a4a-8fd4-4e7e-9590-ed9a52f21059" />

<img width="3012" height="1948" alt="spectral_dim_scaling" src="https://github.com/user-attachments/assets/3500393a-463b-4215-8035-fd1c2282973c" />

<img width="3072" height="1948" alt="gravity_G_scaling" src="https://github.com/user-attachments/assets/9ebcfcb9-0751-40e4-98e9-59ad270125ed" />

<img width="3059" height="1948" alt="cosmological_const_scaling" src="https://github.com/user-attachments/assets/fff01343-ab13-4ff6-8f44-ee9750c76ad6" />

this is a work in progresS—~𓆙𓂀

python main.py -N 12800 -S 2000000 (if you want to snapshot/test/see the universe/structure after 2 million steps and stop there instead of letting it fully crystalize)

<img width="4692" height="2367" alt="N=12800_S=2000000_m" src="https://github.com/user-attachments/assets/c69e06ab-5df0-4e57-bded-1d0466fd935a" />

.. currently simming a N25600 run to hopefully finalize the gravity and other dimensionality curves .. (we might need 2 more x2's to get there maybe even x4)

Next new data point eta >1w

Meanwhile I am working on optimizing the pipeline to try and get to n=1000000 

and then eventually extrapollate trends to a number in between 10^84 and 10^188

<img width="4682" height="2367" alt="N=6400_S=1_m" src="https://github.com/user-attachments/assets/2083c170-de3d-4baf-90cb-ddd55e090b8d" />

<img width="4682" height="2367" alt="N=6400_S=100_m" src="https://github.com/user-attachments/assets/702d4b0d-d432-4a7e-945d-e99f07ea4891" />
<img width="4682" height="2367" alt="N=6400_S=1000_m" src="https://github.com/user-attachments/assets/dda6b4b9-e9c5-4bf2-9c92-4f4cff4d27ca" />
<img width="4682" height="2367" alt="N=6400_S=10000_m" src="https://github.com/user-attachments/assets/4ffc3ba3-cb6b-4d85-8104-1ad29dfddcd7" />
<img width="4682" height="2367" alt="N=6400_S=100000_m" src="https://github.com/user-attachments/assets/0d62bab5-35c0-4243-b90c-b2f0d20ca8e1" />
<img width="4682" height="2367" alt="N=6400_S=200000_m" src="https://github.com/user-attachments/assets/843cff90-b081-48ff-8d7d-95d48ab9a51d" />
<img width="4692" height="2367" alt="N=6400_S=300000_m" src="https://github.com/user-attachments/assets/6d065fb4-e32a-4955-a121-22d43445581e" />
<img width="4690" height="2367" alt="N=6400_S=400000_m" src="https://github.com/user-attachments/assets/36866ed9-efb8-4ea6-ad3e-c86c2b6e33c4" />
<img width="4690" height="2367" alt="N=6400_S=500000_m" src="https://github.com/user-attachments/assets/f1bbc408-65f9-491c-8d87-378ca6a2bb0c" />
<img width="4692" height="2367" alt="N=6400_S=600000_m" src="https://github.com/user-attachments/assets/426396f4-75b2-4b9e-a5c3-128ed6657ff8" />
<img width="4692" height="2367" alt="N=6400_S=700000_m" src="https://github.com/user-attachments/assets/9168f403-c8c5-44ad-a471-866ea463aa95" />
<img width="4692" height="2367" alt="N=6400_S=800000_m" src="https://github.com/user-attachments/assets/ce225a29-fda7-4a81-ab0d-87ef896d4f88" />
<img width="4694" height="2367" alt="N=6400_S=1000000_m" src="https://github.com/user-attachments/assets/4f2b9873-2e27-4127-873b-3ad50ae28c63" />

<img width="4692" height="2367" alt="N=6400_S=2000000_m" src="https://github.com/user-attachments/assets/0c908bbe-6dd5-4b6f-9a8b-0701b273e29f" />



