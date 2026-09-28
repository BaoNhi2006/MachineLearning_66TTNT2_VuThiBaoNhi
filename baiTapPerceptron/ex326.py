def grad(x):
    return 2*x-4
def cost(x):
    return x**2-4*x+5
def myGD1 (x0, eta, n_steps=4):
    x=[x0]
    for it in range(n_steps):
        x_new = x[-1] - eta * grad(x[-1])
        x.append(x_new)
    return (x, n_steps)
(x1, it1) = myGD1(5, 0.2, 4)

# in giá trị ở từng bước (câu 3)
for k, xk in enumerate(x1):
    print('Buoc %d: x = %.4f, f(x) = %.6f' % (k, xk, cost(xk)))

print('Solution x = %f, cost = %f, after %d iteration' % (x1[-1], cost(x1[-1]), it1))
