from micrograd.nn import MLP

n= MLP(3, [4, 4, 1])

xs = [[3.0, 4.0, 1.0],
      [4.0, -1.0, 3.0],
      [2.0, -1.0, 4.0],
      [1.0, -3.0, 6.0]]

ys = [1.0, -1.0, -1.0, 1.0]

ypred = [ n(x) for x in xs ]
Loss = sum((y1 - y2)**2 for y1, y2 in zip(ypred, ys)) 

print("start learning")

for k in range(20):
    ypred = [ n(x) for x in xs ]
    Loss = sum((y1 - y2)**2 for y1, y2 in zip(ypred, ys)) 
    
    for p in n.parameters():
        p.grad =0.0

    Loss.backward()

    for  p in n.parameters():
        p.data += -0.06 * p.grad
    
    
    print(f"The Loss now :{Loss.data} \n ")
