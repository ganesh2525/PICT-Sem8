#include <iostream>
#include <vector>
#include <queue>
#include <stack>
#include <omp.h>

using namespace std;

void sequentialBFS(vector<vector<int>>& graph, int start) {
    int n = graph.size();
    vector<bool> visited(n, false);
    queue<int> q;

    visited[start] = true;
    q.push(start);

    while (!q.empty()) {
        int node = q.front();
        q.pop();

        cout << "Visited node " << node << endl;

        for (int neighbor : graph[node]) {
            if (!visited[neighbor]) {
                visited[neighbor] = true;
                q.push(neighbor);
            }
        }
    }
}

void sequentialDFS(vector<vector<int>>& graph, int start) {
    int n = graph.size();
    vector<bool> visited(n, false);
    stack<int> s;

    s.push(start);

    while (!s.empty()) {
        int node = s.top();
        s.pop();

        if (!visited[node]) {
            visited[node] = true;
            cout << "Visited node " << node << endl;

            for (int neighbor : graph[node]) {
                if (!visited[neighbor]) {
                    s.push(neighbor);
                }
            }
        }
    }
}

void parallelBFS(vector<vector<int>>& graph, int start) {
    int n = graph.size();
    vector<bool> visited(n, false);
    queue<int> q;

    visited[start] = true;
    q.push(start);

    while (!q.empty()) {
        int sz = q.size();

        #pragma omp parallel for
        for (int i = 0; i < sz; i++) {
            int node;
            #pragma omp critical
            {
                node = q.front();
                q.pop();
            }

            printf("Thread %d visited node %d\n", omp_get_thread_num(), node);

            for (int neighbor : graph[node]) {
                if (!visited[neighbor]) {
                    #pragma omp critical
                    {
                        if (!visited[neighbor]) {
                            visited[neighbor] = true;
                            q.push(neighbor);
                        }
                    }
                }
            }
        }
    }
}

void parallelDFS(vector<vector<int>>& graph, int start) {
    int n = graph.size();
    vector<bool> visited(n, false);
    stack<int> s;

    s.push(start);

    while (!s.empty()) {
        int node;
        #pragma omp critical
        {
            node = s.top();
            s.pop();
        }

        if (!visited[node]) {
            visited[node] = true;
            printf("Thread %d visited node %d\n", omp_get_thread_num(), node);

            #pragma omp parallel for
            for (int i = 0; i < graph[node].size(); i++) {
                int neighbor = graph[node][i];
                if (!visited[neighbor]) {
                    #pragma omp critical
                    s.push(neighbor);
                }
            }
        }
    }
}

int main() {
    omp_set_num_threads(4);  // Set number of threads

    vector<vector<int>> graph = {
        {1, 2},    // Node 0
        {0, 3, 4}, // Node 1
        {0, 5},    // Node 2
        {1},       // Node 3
        {1},       // Node 4
        {2}        // Node 5
    };

    // Measure sequential BFS time using omp_get_wtime
    double start_time = omp_get_wtime();
    cout << "Sequential BFS starting from node 0:\n";
    sequentialBFS(graph, 0);
    double end_time = omp_get_wtime();
    double seq_bfs_time = end_time - start_time;
    cout << "Sequential BFS execution time: " << seq_bfs_time << " seconds\n";

    // Measure sequential DFS time using omp_get_wtime
    start_time = omp_get_wtime();
    cout << "\nSequential DFS starting from node 0:\n";
    sequentialDFS(graph, 0);
    end_time = omp_get_wtime();
    double seq_dfs_time = end_time - start_time;
    cout << "Sequential DFS execution time: " << seq_dfs_time << " seconds\n";

    // Measure parallel BFS time using omp_get_wtime
    start_time = omp_get_wtime();
    cout << "\nParallel BFS starting from node 0:\n";
    parallelBFS(graph, 0);
    end_time = omp_get_wtime();
    double par_bfs_time = end_time - start_time;
    cout << "Parallel BFS execution time: " << par_bfs_time << " seconds\n";

    // Measure parallel DFS time using omp_get_wtime
    start_time = omp_get_wtime();
    cout << "\nParallel DFS starting from node 0:\n";
    parallelDFS(graph, 0);
    end_time = omp_get_wtime();
    double par_dfs_time = end_time - start_time;
    cout << "Parallel DFS execution time: " << par_dfs_time << " seconds\n";

    return 0;
}
