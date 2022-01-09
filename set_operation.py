N={1,2,3,4,5,6,7,8,9}
E={2,4,6,8,10}
#union of N and E-
Union=N.union(E)
print("Union of N and E is",Union)
#intersection of N and E-
Intersection=N.intersection(E)
print("Intersection of N and E is",Intersection)
#Difference of N and E-
Difference=N.difference(E)
print("Difference of N and E is",Difference)
#symmetric difference of N and E-
d=E.difference(N)
Sym_difference=Difference.union(d)
print("Symmetric difference of N and E is",Sym_difference)
