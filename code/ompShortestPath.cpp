/* IT 388/487
 * OMP parallel implementation of Johnson's algorithm
 *
 * Compile with: g++ -g -o omp ompShortestPath.cpp -fopenmp -O3
 * <<< The -O3 flag is an optimization flag to improve performance >>>
 * 
 * Execute with ./omp <# Threads> <input filename> <[1|0] display progress in console>
 * 
 * Authors: Nick Kolesar, Aaron Sihweil
 */
#include <iostream>
#include <fstream>
#include <omp.h>
#include <vector>
#include <limits>
#include <chrono>

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
    vector<int> new_dist(V + 1);

    // Add a new source vertex to the graph and connect it to all original vertices with 0 weight edges
    vector<vector<int>> edges_with_extra(edges);
    for (int i = 0; i < V; ++i) {
        edges_with_extra.push_back({V, i, 0});
    }

    // Relax all edges |V| - 1 times
    for (int i = 0; i < V; ++i) {
        new_dist = dist;

        #pragma omp parallel for
        for (int j = 0; j < (int)edges_with_extra.size(); ++j) {
            auto &edge = edges_with_extra[j];
            if (dist[edge[0]] != INF && dist[edge[0]] + edge[2] < new_dist[edge[1]]) {
                #pragma omp critical
                new_dist[edge[1]] = dist[edge[0]] + edge[2];
            }
        }

        dist.swap(new_dist);
    }
    return vector<int>(dist.begin(), dist.begin() + V);  // Return distances excluding the new source vertex
}


void JohnsonAlgorithm(const vector<vector<int>>& graph, const bool display_progress = false) {
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
    if(V <= 50){
        cout << "Modified Graph:\n";
        for (const auto& row : altered_graph) {
            for (int weight : row) {
                cout << weight << ' ';
            }
            cout << endl;
        }
    }
    
    vector<vector<int>> all_distances(V, vector<int>(V, INF));
    
    
    int verticesCompleted = 0; // shared counter for displaying progress


    // Run Dijkstra's algorithm for every vertex as the source
    #pragma omp parallel for
    for (int source = 0; source < V; ++source) {
        Dijkstra_Algorithm(graph, altered_graph, source, all_distances);
        if (display_progress)
            #pragma omp critical
                cout << "\rProgress: " << ++verticesCompleted << " / " << V << " vertices completed." << flush;
        
    }

    // add new line after dijkstra progress completion
    cout << endl;
    
    // Print all shortest distances
    if(V <= 50){
        for (int source = 0; source < V; source++)
        printShortestDistances(source, all_distances[source]);
    }
}

void readGraph(ifstream& infile, vector<vector<int>>& graph) {
    int numFromVertices, numToVertices, numEdges, weight;

    infile >> numFromVertices >> numToVertices >> numEdges;
    cout << "Reading " << numFromVertices << " x " << numToVertices << " graph with " << numEdges << " edges." << endl;

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

// Function to hide the cursor in the console (linux only)
void hideCursor() {
    cout << "\033[?25l";
}

// Function to show the cursor in the console (linux only)
void showCursor() {
    cout << "\033[?25h";
}

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        cerr << "Usage: " << argv[0] << " <# Threads> <input_file> [1|0 for displaying progress]\n";
        return 1;
    }

    int num_threads = stoi(argv[1]);
    ifstream infile(argv[2]);

    bool display_progress = false;
    // Optional argument to display progress since it slows down execution
    if (argc > 3)
    {
        display_progress = stoi(argv[3]) != 0;
    }

    if (!infile)
    {
        cerr << "Error opening file: " << argv[2] << endl;
        return 1;
    }

    // Set the number of threads for OpenMP
    omp_set_num_threads(num_threads);

    // Define the graph
    vector<vector<int>> graph;

    hideCursor();

    // Read the graph from the input file
    readGraph(infile, graph);

    // Execute Johnson's Algorithm
    auto start = chrono::high_resolution_clock::now();
    JohnsonAlgorithm(graph, display_progress);
    auto end = chrono::high_resolution_clock::now();

    showCursor();

    chrono::duration<double> elapsed = end - start;
    cout << "Elapsed time: " << elapsed.count() << " seconds\n";
    return 0;
}