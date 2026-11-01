import csv

rows = list(csv.DictReader(open("state.csv")))
mj = [int(r["matched"]) for r in rows]

runs = []
cur, n = mj[0], 1
for v in mj[1:]:
    if v == cur:
        n += 1
    else:
        runs.append((cur, n)); cur, n = v, 1
runs.append((cur, n))

print("总段数:", len(runs))
print("前 12 段:", runs[:12])
print("后 12 段:", runs[-12:])
lens = [n for _, n in runs]
print("段长分布: min=%d  max=%d  median=%.1f" % (min(lens), max(lens), sorted(lens)[len(lens)//2]))