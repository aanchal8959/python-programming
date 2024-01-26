import random
def game(comp,you):
    if comp==you:
        return None
    elif comp=='r':
        if you=='p':
            return True
        elif you=='s':
            return False
    elif comp=='p':
        if you=='r':
            return False
        elif you=='s':
            return True
    elif comp=='s':
        if you=='p':
            return False
        elif you=='r':
            return True
print("comp turn: rock(r), paper(p), scissor(s)")
randno=random.randint(1,3)
if randno==1:
    comp='r'
elif randno==2:
    comp='p'
else:
    comp='s'
you=input("player's turn: rock(r), paper(p), scissor(s)")
a=game(comp,you)
print(f"comp choose: {comp}")
print(f"you choose: {you}")
if a==None:
    print("The game is tie!")
elif a:
    print("you win!")
else:
    print("you lose!")