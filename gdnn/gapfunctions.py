import numpy as np
from gappy import gap


@gap.gap_function
def group_from_generators(generators):
	"""
	function(generators)
	local G;
	generators := List(generators, PermList);
	G := GroupWithGenerators(generators);
	return G;
	end;
	"""


@gap.gap_function
def subgroup_pairs_HK(G, indices):
	"""
	function(G, indices)
	local H, Hs, K, Ks, K_classes, NH, NK, out;
	out := [];
	Hs := Filtered(LowIndexSubgroups(G, Maximum(indices)), H->(Index(G, H) in indices));
	for H in Hs do
		NH := Normalizer(G, H);
		K_classes := Filtered(LowIndexSubgroups(H, 2), K->Order(K)<Order(H));
		Ks := [H];
		for K in K_classes do
			NK := Normalizer(NH, K);
			Append(Ks, List(RightTransversal(NH, NK), g->ConjugateGroup(K, g)));
		od;
		Add(out, rec(H:=H, Ks:=Ks));
	od;
	return out;
	end;
	"""


@gap.gap_function
def theta(G, H, K, J):
	"""
	function(G, H, K, J)
	local sortfunc, actfunc, double_cosets, h, omega, x, xs, S;

	sortfunc := function(omega)
		local w;
		for w in omega do
			Sort(w);
		od;
		return omega;
	end;

	actfunc := function(omega, g)
		local omega_g;
		omega_g := sortfunc( List(omega, w->List(w, x->CanonicalRightCosetElement(J, x*g))) );
		return omega_g;
	end;

	if Index(H, K) = 2 then
		for x in RightTransversal(H, K) do
			if not x in K then
				h := x;
			fi;
		od;
		xs := List(DoubleCosetRepsAndSizes(G, J, K), L->L[1]);
		double_cosets := List(Filtered(List(xs, x->[x, DoubleCoset(J, x, K)]), L->not (L[1]*h in L[2])), L->L[2]);
	else
		double_cosets := DoubleCosets(G, J, K);
	fi;

	omega := sortfunc( List(double_cosets, D->ShallowCopy(RepresentativesContainedRightCosets(D))) );
	#S := Stabilizer(G, omega, actfunc);
	S := Group(Filtered(Elements(G), g->actfunc(omega, g)=omega));
	return S;
	end;
	"""


@gap.gap_function
def permgroup_decomposition(G, Gamma, degree):
	"""
	function(G, Gamma, degree)
	local g, hom, J, os, out, perm, perms, perms_g, pnts, post_stabilizers, transversal;

	pnts := [1..degree];
	post_stabilizers := [];
	while Length(pnts) > 0 do
		os := OrbitStabilizer(Gamma, Minimum(pnts));
		Add(post_stabilizers, os.stabilizer);
		pnts := Filtered(pnts, x->not x in os.orbit);
	od;

	hom := GroupHomomorphismByImages(G, Gamma);
	out := rec();
	out.stabilizers := List(post_stabilizers, J->PreImage(hom, J));

	perms := [];
	for g in GeneratorsOfGroup(G) do
		perms_g := [];
		for J in out.stabilizers do
			transversal := RightTransversal(G, J);
			perm := List(transversal, x->PositionCanonical(transversal, x*g));
			if Length(perms_g) >= 1 then
				perm := perm + Maximum(Last(perms_g));
			fi;
			Add(perms_g, perm);
		od;
		perms_g := Concatenation(perms_g);
		Add(perms, perms_g);
	od;

	perms	 := List(perms, PermList);
	hom := GroupHomomorphismByImages(GroupWithGenerators(perms), Gamma);
	out.conjugator := ListPerm(Inverse(ConjugatorOfConjugatorIsomorphism(hom)), degree);
	return out;
	end;
	"""


@gap.gap_function
def subgroup_generators(G, H):
	"""
	function(G, H)
	local generators;
	generators := List(GeneratorsOfGroup(H), g->Factorization(G, g));
	return generators;
	end;
	"""


def subgroup_generators_str(G, H, generator_names):
	generators = [g.__repr__() for g in subgroup_generators(G, H).python()]
	out = "["+", ".join(generators)+"]"
	for (i, name) in enumerate(generator_names, start=1):
		out = out.replace(f"x{i:d}", name)
	return out


@gap.gap_function
def weight_pattern_J(G, H, K, J):
	"""
	function(G, H, K, J)
	local color, colors, coset_rep_sets, coset_reps, double_cosets, h, i, muls, pattern, pair, pairs, perm, positions, positions_h, transversal_H, transversal_J, x, xs;

	if Index(H, K) = 2 then

		for x in RightTransversal(H, K) do
			if not x in K then
				h := x;
			fi;
		od;
		xs := List(DoubleCosetRepsAndSizes(G, J, K), L->L[1]);
		double_cosets := List(xs, x->DoubleCoset(J, x, K));
		positions := [1..Length(double_cosets)];
		positions_h := List(xs, x->Position(double_cosets, DoubleCoset(J, x*h, K)));
		pairs := Filtered(List(positions, i->[i, positions_h[i]]), L->L[1]<L[2]);
		colors := ListWithIdenticalEntries(Length(double_cosets), 0);
		for i in [1..Length(pairs)] do
			pair := pairs[i];
			colors[pair[1]] := i;
			colors[pair[2]] := -i;
		od;

	else
		double_cosets := DoubleCosets(G, J, K);
		colors := [1..Length(double_cosets)];
	fi;

	coset_rep_sets := List(double_cosets, D->RepresentativesContainedRightCosets(D));
	muls := List(coset_rep_sets, Length);
	colors := Concatenation(ListN(muls, colors, ListWithIdenticalEntries));

	coset_reps := Concatenation(coset_rep_sets);
	transversal_J := RightTransversal(G, J);
	perm := PermList( List(coset_reps, g->PositionCanonical(transversal_J, g)) );
	colors := Permuted(colors, perm);

	transversal_H := RightTransversal(G, H);
	pattern := List(transversal_H, g->Permuted(colors, PermList(List(transversal_J, x->PositionCanonical(transversal_J, x*g)))));
	return pattern;
	end;
	"""


def weight_pattern(G, H, K, Js):
	patterns = [np.array(weight_pattern_J(G, H, K, J).python()) for J in Js]
	shifted = []
	for pattern in patterns:
		if len(shifted) == 0:
			shifted.append(pattern)
			continue
		shifted.append( pattern + np.sign(pattern)*np.max(shifted[-1]) )
	pattern = np.concatenate(shifted, 1)
	return pattern


class SubgroupPairsHK(object):

	def __init__(self, G, indices):
		self.subgroup_pairs = subgroup_pairs_HK(G, indices)

	def HKs(self):
		for HK in self.subgroup_pairs:
			H = HK["H"]
			Ks = HK["Ks"]
			for K in Ks:
				yield (H, K)

	def Hs(self):
		subgroups = [HK["H"] for HK in self.subgroup_pairs]
		for H in subgroups:
			yield H

	def Ks(self):
		for HK in self.subgroup_pairs:
			H = HK["H"]
			Ks = HK["Ks"]
			for K in Ks:
				yield K
