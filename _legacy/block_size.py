import matplotlib.pyplot as plt

n = 34

threads = []
for k in range(34):
    i = k & ~1

    currNumElements = n - k - 1
    currThreads = currNumElements * i

    threads.append(currThreads)
    print(f'k = {k}, i = {i}, threads = {currThreads}')

plt.plot(range(n), threads, marker='o')
plt.xlabel('k')
plt.ylabel('Number of Threads')
plt.title(f'Threads vs k (n={n})')
plt.grid(True)

plt.savefig('threads_vs_k.png', dpi=300)