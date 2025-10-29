#include <iostream>
#include <fstream>
#include <omp.h>
#include <vector>
#include <algorithm>
#include <limits>

#define INF std::numeric_limits<int>::max()

using namespace std;

int Min_Distance(const vector<int>& dist, const vector<bool>& visited) {
    int min = INF, min_index;
    for (int v = 0; v < dist.size(); ++v) {
        if (!visited[v] && dist[v] <= min) {
            min = dist[v];
            min_index = v;
        }
    }
    return min_index;
}

void printShortestDistances(int source, const vector<int>& dist) {
    int V = dist.size();
    cout << "\nShortest Distance with vertex " << source << " as the source:\n";
    cout << "Shortest Distance from vertex " << source << ":" << endl;
    for (int i = 0; i < V; ++i) {
        cout << "Vertex " << i << ": " << (dist[i] == INF ? "INF" : to_string(dist[i])) << endl;
    }
}

void Dijkstra_Algorithm(const vector<vector<int>>& graph, const vector<vector<int>>& altered_graph, int source, vector<vector<int>>& all_distances) {
    int V = graph.size();  // Number of vertices
    vector<int> dist(V, INF);  // Distance from source to each vertex
    vector<bool> visited(V, false);  // Track visited vertices
    
    dist[source] = 0;  // Distance to source itself is 0

    
    for (int count = 0; count < V - 1; ++count) {
        // Select the vertex with the minimum distance that hasn't been visited
        int u = Min_Distance(dist, visited);
        visited[u] = true;  // Mark this vertex as visited

        // Update the distance value of the adjacent vertices of the selected vertex
        for (int v = 0; v < V; ++v) {
            if (!visited[v] && graph[u][v] != 0 && dist[u] != INF && dist[u] + altered_graph[u][v] < dist[v]) {
                dist[v] = dist[u] + altered_graph[u][v];
            }
        }
    }

    all_distances[source] = dist;
}


vector<int> BellmanFord_Algorithm(const vector<vector<int>>& edges, int V) {
    vector<int> dist(V + 1, INF);  // Distance from source to each vertex
    dist[V] = 0;  // Distance to the new source vertex (added vertex) is 0

    // Add a new source vertex to the graph and connect it to all original vertices with 0 weight edges
    vector<vector<int>> edges_with_extra(edges);
    for (int i = 0; i < V; ++i) {
        edges_with_extra.push_back({V, i, 0});
    }

    // Relax all edges |V| - 1 times
    for (int i = 0; i < V; ++i) {
        for (const auto& edge : edges_with_extra) {
            if (dist[edge[0]] != INF && dist[edge[0]] + edge[2] < dist[edge[1]]) {
                dist[edge[1]] = dist[edge[0]] + edge[2];
            }
        }
    }
    return vector<int>(dist.begin(), dist.begin() + V);  // Return distances excluding the new source vertex
}


void JohnsonAlgorithm(const vector<vector<int>>& graph) {
    int V = graph.size();  // Number of vertices
    vector<vector<int>> edges;
    
    // Collect all edges from the graph
    for (int i = 0; i < V; ++i) {
        for (int j = 0; j < V; ++j) {
            if (graph[i][j] != 0) {
                edges.push_back({i, j, graph[i][j]});
            }
        }
    }

    // Get the modified weights from Bellman-Ford algorithm
    vector<int> altered_weights = BellmanFord_Algorithm(edges, V);
    vector<vector<int>> altered_graph(V, vector<int>(V, 0));

    // Modify the weights of the edges to remove negative weights
    for (int i = 0; i < V; ++i) {
        for (int j = 0; j < V; ++j) {
            if (graph[i][j] != 0) {
                altered_graph[i][j] = graph[i][j] + altered_weights[i] - altered_weights[j];
            }
        }
    }

    // Print the modified graph with re-weighted edges
    cout << "Modified Graph:\n";
    for (const auto& row : altered_graph) {
        for (int weight : row) {
            cout << weight << ' ';
        }
        cout << endl;
    }

    vector<vector<int>> all_distances(V, vector<int>(V, INF));

    // Run Dijkstra's algorithm for every vertex as the source
    #pragma omp parallel for
    for (int source = 0; source < V; ++source) {
        Dijkstra_Algorithm(graph, altered_graph, source, all_distances);
    }

    // Print all shortest distances
    for (int source = 0; source < V; source++)
        printShortestDistances(source, all_distances[source]);
}

void readGraph(ifstream& infile, vector<vector<int>>& graph) {
    int numFromVertices, numToVertices, numEdges, weight;

    infile >> numFromVertices >> numToVertices >> numEdges;
    cout << "Reading graph with " << numFromVertices << " vertices, " << numToVertices << " vertices, and " << numEdges << " edges." << endl;

    // Initialize the graph with zeros
    for (int i = 0; i < numFromVertices; i++)
    {
        vector<int> row(numToVertices, 0);
        graph.push_back(row);
    }

    // Read edges and populate the graph
    while (infile >> numFromVertices >> numToVertices >> weight)
    {
        graph[numFromVertices][numToVertices] = weight;
    }
}

void printGraph(const vector<vector<int>>& graph) {
    cout << "Graph adjacency matrix:\n";
    for (const auto& row : graph) {
        for (int weight : row) {
            cout << weight << ' ';
        }
        cout << endl;
    }
}

int main(int argc, char** argv)
{
    if (argc != 3)
    {
        cerr << "Usage: " << argv[0] << " <# Threads> <input_file>\n";
        return 1;
    }

    int num_threads = stoi(argv[1]);
    ifstream infile(argv[2]);

    if (!infile)
    {
        cerr << "Error opening file: " << argv[2] << endl;
        return 1;
    }

    // Set the number of threads for OpenMP
    omp_set_num_threads(num_threads);

    // Define the graph
    vector<vector<int>> graph;

    // Read the graph from the input file
    readGraph(infile, graph);

    // Execute Johnson's Algorithm
    JohnsonAlgorithm(graph);
    return 0;
}