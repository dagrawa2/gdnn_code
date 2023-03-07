import os
import pickle
import gdnn

from gappy import gap


def outersect_idx2subgroups(G, H_1, H_2):
	K = gap.Intersection(H_1, H_2)
	for x in gap.RightTransversal(G, H_1):
		if x not in H_1:
			g_1 = x
			break
	for x in gap.RightTransversal(G, H_2):
		if x not in H_2:
			g_2 = x
			break
	g = gap.Random( gap.Intersection(gap.RightCoset(H_1, g_1), gap.RightCoset(H_2, g_2)) )
	H = gap.ClosureGroup(K, g)
	return H, K

def group_generators(H):
	generators = [gap.ListPerm(g, 32).python() for g in H.GeneratorsOfGroup()]
	return generators


with open("data/generators.pkl", "rb") as f:
	generators = pickle.load(f)

generators = [g for (_, _, g) in generators]
G = gdnn.gapfunctions.group_from_generators(generators)

reps_1 = []
for i in range(8):
	v = [0 for _ in range(32)]
	v[4*i+1] = 1
	v[4*i+3] = 1
	K = gap.Stabilizer(G, v, gap.Permuted)
	K_generators = group_generators(K)
	H_generators = [g for g in K_generators]
	h = [i+1 for i in range(32)]
	h[4*i+0] += 1
	h[4*i+1] -= 1
	h[4*i+2] += 1
	h[4*i+3] -= 1
	H_generators.append(h)
	reps_1.append( (H_generators, K_generators) )

reps_2 = []
Hs = [gdnn.gapfunctions.group_from_generators(H) for (H, K) in reps_1]
for (H_1, H_2) in zip(Hs[:-1:2], Hs[1::2]):
	H, K = outersect_idx2subgroups(G, H_1, H_2)
	reps_2.append( (group_generators(H), group_generators(K)) )

reps_3 = []
Hs = [gdnn.gapfunctions.group_from_generators(H) for (H, K) in reps_2]
for (H_1, H_2) in zip(Hs[:-1:2], Hs[1::2]):
	H, K = outersect_idx2subgroups(G, H_1, H_2)
	reps_3.append( (group_generators(H), group_generators(K)) )

reps_4 = []
Hs = [gdnn.gapfunctions.group_from_generators(H) for (H, K) in reps_3]
for (H_1, H_2) in zip(Hs[:-1:2], Hs[1::2]):
	H, K = outersect_idx2subgroups(G, H_1, H_2)
	reps_4.append( (group_generators(H), group_generators(K)) )

#reps_4 = [(reps_4[0][0], reps_4[0][0])] + reps_4

reps_5 = [(generators, generators)]

reps = [reps_1, reps_2, reps_3, reps_4, reps_5]

with open("data/reps.pkl", "wb") as f:
	pickle.dump(reps, f)

print("Done!")
