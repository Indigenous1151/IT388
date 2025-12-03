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
#include <optional>
#include <fstream>
#include <chrono>
#include <limits>
#include <vector>
#include <queue>
#include <omp.h>
#include <tuple>
#include <list>

#define INF std::numeric_limits<double>::max()

struct Edge {
    int toVertex;
    double weight;

    Edge(int to, double weight) : toVertex(to), weight(weight) {}
};

// Alias for vector<list<T>> because it's annoying to write
template<typename T>
using AdjList = std::vector<std::list<T>>;
// Another alias for vector<vector<T>> for the same reason
template<typename T>
using AdjMatrix = std::vector<std::vector<T>>;

using namespace std;

// Prototypes
vector<double> Dijkstra_Algorithm(const AdjList<Edge>&, int);
optional<AdjMatrix<double>> JohnsonAlgorithm(const AdjList<Edge>&, const bool);
optional<vector<double>> BellmanFord_Algorithm(const AdjList<Edge>&, int);
tuple<double, double, double, int> getStats(const AdjMatrix<double>&);
int Min_Distance(const vector<int>&, const vector<bool>&);
void printShortestDistances(int, list<Edge>&);
void printResults(ostream&, const AdjMatrix<double>&);
void readGraph(ifstream&, AdjList<Edge>&, bool);
void printGraph(const AdjList<Edge>&);
void hideCursor();
void showCursor();

// Function to find the vertex with the minimum distance value
int Min_Distance(const vector<double>& dist, const vector<bool>& visited) {
    double min = INF;
    int min_index;
    for (int v = 0; v < dist.size(); ++v) {
        if (!visited[v] && dist[v] <= min) {
            min = dist[v];
            min_index = v;
        }
    }
    return min_index;
}

// Function to print shortest distances from source
void printShortestDistances(int source, list<Edge>& dist) {
    int V = dist.size();
    cout << "\nShortest Distance with vertex " << source << " as the source:\n";
    cout << "Shortest Distance from vertex " << source << ":" << endl;

    // Defining the for loop variables before the loop beacuse they are different types
    int i = 0;
    list<Edge>::iterator it = dist.begin();
    for (; it != dist.end(); it++, i++)
        cout << "Vertex " << i << ": " << (it->weight == INF ? "INF" : to_string(it->weight)) << endl;
}

// Dijkstra's algorithm implementation using a priority queue
vector<double> Dijkstra_Algorithm(const AdjList<Edge>& graph, int source) {
    int V = graph.size();
    vector<double> dist(V, INF);
    vector<bool> visited(V, false);
    dist[source] = 0;

    // Min distance priority queue
    using P = pair<double, int>; // pair -> {distance, vertex}
    priority_queue<P, vector<P>, greater<P>> pq;
    pq.push({0, source});

    while (!pq.empty())
    {
        auto [d, u] = pq.top();
        pq.pop();
        if (visited[u]) continue;
        visited[u] = true;

        for (const Edge& e : graph[u])
        {
            int v = e.toVertex;
            double w = e.weight;
            if (!visited[v] && d + w < dist[v])
            {
                dist[v] = d + w;
                pq.push({dist[v], v});
            }
        }
    }

    return dist;
}

// Bellman-Ford algorithm implementation
std::optional<vector<double>> BellmanFord_Algorithm(const AdjList<Edge>& graph, int source) {
    int V = graph.size();
    vector<double> dist(V, INF);
    dist[source] = 0;

    // Create an edge list
    vector<tuple<int,int,double>> edges; // {from, to, weight}
    for (int u = 0; u < V; u++)
        for (const Edge& e : graph[u])
            edges.push_back({u, e.toVertex, e.weight});

    // Relax edges V-1 times
    for (int i = 0; i < V - 1; i++)
        for (auto [u, v, w] : edges)
            if (dist[u] != INF && dist[u] + w < dist[v])
                dist[v] = dist[u] + w;

    // Detect negative cycles
    for (auto [u, v, w] : edges)
    {
        if (dist[u] != INF && dist[u] + w < dist[v])
        {
            return std::nullopt; // tells optional there was a problem
        }
    }

    return dist;
}
// Johnson's algorithm implementation
optional<AdjMatrix<double>> JohnsonAlgorithm(AdjList<Edge>& graph, const bool display_progress = false) {
    int V = graph.size();

    auto stepOneStart = chrono::high_resolution_clock::now();
    // Step 1: add a new vertex connected to all others with 0-weight edges
    // This guarantees that Bellman-Ford has access to all vertices
    AdjList<Edge> extendedGraph = graph;
    extendedGraph.push_back({});
    for (int v = 0; v < V; v++)
        extendedGraph[V].push_back({v, 0});

    auto stepOneEnd = chrono::high_resolution_clock::now();

    auto stepTwoStart = chrono::high_resolution_clock::now();
    // Step 2: run Bellman-Ford from the new vertex to get h(v)
    // h(v) is the shortest path from the extended row to v and
    // serves as a finite offset for each vertex
    auto bellman = BellmanFord_Algorithm(extendedGraph, V);

    if (!bellman)
    {
        return std::nullopt; // propagate the nullopt
    }

    const auto& h = *bellman; // BellmanFord Successful

    auto stepTwoEnd = chrono::high_resolution_clock::now();

    auto stepThreeStart = chrono::high_resolution_clock::now();
    // Step 3: reweight all edges
    // This step gets rid of all negative weights by offsetting by h(v)
    AdjList<Edge> reweightedGraph(V);
    #pragma omp parallel for schedule(dynamic)
    for (int u = 0; u < V; u++)
    {
        for (const Edge& e : graph[u])
        {
            double newWeight = e.weight + h[u] - h[e.toVertex];
            reweightedGraph[u].push_back({e.toVertex, newWeight});
        }
    }

    auto stepThreeEnd = chrono::high_resolution_clock::now();

    auto stepFourStart = chrono::high_resolution_clock::now();
    // Step 4: run Dijkstra from each vertex
    // Standard priority queue based dijkstra's implementation
    // run in a for loop across the entire graph
    int verticesCompleted = 0; // progress display variable
    AdjMatrix<double> distanceMatrix(V, vector<double>(V, INF));
    // Parallelize with dynamic scheduling because adjacency lists are not consistent lengths
    #pragma omp parallel for schedule(dynamic)
    for (int u = 0; u < V; u++)
    {
        vector<double> dist = Dijkstra_Algorithm(reweightedGraph, u);
        for (int v = 0; v < V; v++)
        {
            if (dist[v] != INF)
                // Step 5: Get original weights
                distanceMatrix[u][v] = dist[v] - h[u] + h[v];
        }

        // Allow user to see progress of the program when display_progress is true
        if (display_progress)
        {
            #pragma omp atomic
            verticesCompleted++;
            if (omp_get_thread_num() == 0)
                cout << "\rProgress: [" << verticesCompleted << "/" << V << "] vertices completed." << flush;
        }
    }

    auto stepFourEnd = chrono::high_resolution_clock::now();
    if (display_progress)
        cout << "\rProgress: [" << verticesCompleted << "/" << V << "] vertices completed." << endl;

    // Display step times
    chrono::duration<double> stepOne = stepOneEnd - stepOneStart;
    chrono::duration<double> stepTwo = stepTwoEnd - stepTwoStart;
    chrono::duration<double> stepThree = stepThreeEnd - stepThreeStart;
    chrono::duration<double> stepFour = stepFourEnd - stepFourStart;


    cout << "Add Extra Vertex Elapsed Time:     " << stepOne.count()   << " Seconds\n"
         << "BellMan-Ford Elapsed Time:         " << stepTwo.count()   << " Seconds\n"
         << "Reweight Edges Elapsed Time:       " << stepThree.count() << " Seconds\n"
         << "Dijkstra/Fix weights Elapsed Time: " << stepFour.count()  << " Seconds" << endl;
    // return an adjacencyMatrix for all distances
    return distanceMatrix;
}

// Function to print the results or export them to a file
void printResults(ostream& output, const AdjMatrix<double>& graph) {

    tuple<double, double, double, int> stats = getStats(graph);
    long graphSize = graph.size() * graph[0].size();

    output << endl;
    output << "Longest Distance: " << get<0>(stats) << endl;
    output << "Shortest Non-Zero Distance: " << get<1>(stats) << endl;
    output << "Average Distance: " << get<2>(stats) << endl;
    output << "INF Distance count: " << get<3>(stats) << '/' << graphSize << endl;
}

// Function to compute statistics about the shortest path distances
tuple<double,double,double,int> getStats(const AdjMatrix<double>& graph)
{
    int rows = graph.size();
    int cols = graph[0].size();
    long graphTotalSize = (long)rows * cols;

    // set max and min to extremes
    double maxVal = std::numeric_limits<double>::min();
    double minNonZero = std::numeric_limits<double>::max();
    long double total = 0;
    int numValidDistances = 0;
    int numINF = 0;

    #pragma omp parallel for reduction(max:maxVal) \
                             reduction(min:minNonZero) \
                             reduction(+:total,numValidDistances,numINF) \
                             schedule(static)
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            double cur = graph[i][j];
            if (cur == INF)
                ++numINF;
            else if (cur != 0)
            {
                if (cur > maxVal)
                    maxVal = cur;
                if (cur < minNonZero)
                    minNonZero = cur;
                total += cur;
                numValidDistances++;
            }
        }
    }

    double average = (numValidDistances > 0) ? total / numValidDistances : 0.0;

    // handle when there are no valid distances
    if (numValidDistances == 0)
    {
        minNonZero = INF;
        maxVal = INF;
    }

    return make_tuple(maxVal, minNonZero, average, numINF);
}

// Function to read the graph from an input file
void readGraph(ifstream& infile, AdjList<Edge>& graph, bool display_progress = false) {
    int from, to, numEdges;
    double weight;

    infile >> from >> to >> numEdges;
    cout << "Reading " << from << " x " << to << " graph with " << numEdges << " edges." << endl;

    // Initialize the adjacency list graph
    graph.assign(from, list<Edge>());

    // Read edges and populate the graph
    while (infile >> from >> to >> weight)
    {
        static int count = 1;
        graph[from].push_back(Edge(to, weight));
        if (display_progress)
            cout << "\rEdges read: " << count++ << flush;
    }

    if (display_progress)
        cout << endl;
}

// Function to print the graph
void printGraph(const AdjList<Edge>& graph) {
    cout << "Graph adjacency list:\n";
    for (const list<Edge>& row : graph)
    {
        for (const Edge& edge : row)
            cout << edge.weight << ' ';
        cout << endl;
    }
}

// Function to hide the cursor in the console (linux only)
void hideCursor() { cout << "\033[?25l"; }

// Function to show the cursor in the console (linux only)
void showCursor() { cout << "\033[?25h"; }


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
    AdjList<Edge> graph;

    hideCursor();

    // Read the graph from the input file
    readGraph(infile, graph, display_progress);

    // Flush the output buffer to ensure the progress bar works well
    cout << flush;

    // Execute Johnson's Algorithm
    auto start = chrono::high_resolution_clock::now();
    optional<AdjMatrix<double>> all_distances_opt = JohnsonAlgorithm(graph, display_progress);
    // fail gracefully if negative-weight cycle
    if (!all_distances_opt)
    {
        cerr << "Graph contains a negative-weight cycle!\n";
        showCursor();
        return 1;
    }

    const auto& all_distances = *all_distances_opt;
    auto end = chrono::high_resolution_clock::now();

    showCursor();

    chrono::duration<double> elapsed = end - start;
    cout << "Total Elapsed time: " << elapsed.count() << " seconds\n";

    // Print or export results
    printResults(cout, all_distances);

    return 0;
}
