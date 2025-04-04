@echo off

call .venv/Scripts/activate

set locations=970 2000 2200 2820 2825 2827 2846 3001 3002 3120 3122 3126 3127 3180 3662 3682 3685 3804 3812 4030 4032 4034 4035 4040 4043 4051 4057 4063 4262 4263 4264 4266 4270 4272 4273 4321 4324 4335 4812 4821
set models= lstm gru saes xgboost

for %%l in (%locations%) do (
    for %%m in (%models%) do (
        echo Running script with %%l and %%m
        python train.py --model %%m --location %%l
    )
)

echo training completed!
pause

