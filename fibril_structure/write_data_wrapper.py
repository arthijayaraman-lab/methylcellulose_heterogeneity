import subprocess

trials = [1, 2, 3]
dss = [1.8, 2.0, 2.4]
htypes = ['c','s']
hs = {'c': [0,0.5, 1],
      's': [1, 4, 7]}

for trial in trials:
    for ds in dss:
        for ht in htypes:
            hvals = hs[ht]
            for h in hvals:
                runstr = ['python3', 'write_data_mc.py']
                runstr.extend([str(int(trial)), ht, str(h), str(ds)])
                print(runstr)
                subprocess.call(runstr)
