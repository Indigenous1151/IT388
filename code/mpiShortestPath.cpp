/* IT 388/487
 * MPI parallel implementation of Johnson's algorithm
 *
 * Compile with: mpic++ -std=c++17 -g -o mpi mpiShortestPath.cpp -O3
 * <<< The -O3 flag is an optimization flag to improve performance >>>
 * <<< The -std=c++17 flag specifies the compiler to use C++17 >>>
 *
 * Execute with mpiexec -np <# Threads> ./mpi <input filename> <[1|0] display progress in console>
 *
 * Authors: Nick Kolesar, Aaron Sihweil, Jordan Davis, Ryan Kelly
 */
#include <iostream>
#include <optional>
#include <fstream>
#include <chrono>
#include <limits>
#include <vector>
#include <queue>
#include <mpi.h>
#include <tuple>
#include <list>

#define INF std::numeric_limits<int>::max()

struct Edge {
    int toVertex;
    int weight;

    Edge(int to, int weight) : toVertex(to), weight(weight) {}
};

// Alias for vector<list<T>> because it's annoying to write
template<typename T>
using AdjList = std::vector<std::list<T>>;
// Another alias for vector<vector<T>> for the same reason
template<typename T>
using AdjMatrix = std::vector<std::vector<T>>;

using namespace std;

// Prototypes
vector<int> Dijkstra_Algorithm(const AdjList<Edge>&, int);
optional<AdjMatrix<int>> JohnsonAlgorithm(const AdjList<Edge>&, const bool);
optional<vector<int>> BellmanFord_Algorithm(const AdjList<Edge>&, int);
tuple<int, int, double, int> getStats(const AdjMatrix<int>&);
int Min_Distance(const vector<int>&, const vector<bool>&);
void printShortestDistances(int, const vector<int>&);
void printResults(ostream&, const AdjMatrix<int>&);
void readGraph(ifstream&, AdjList<Edge>&, bool);
void printGraph(const AdjList<Edge>&);
void hideCursor();
void showCursor();
void broadcastGraph(AdjList<Edge>& graph, int rank);

// Function to find the vertex with the minimum distance value
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

// Function to print the shortest distances from the source vertex
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
std::vector<int> Dijkstra_Algorithm(const AdjList<Edge>& graph, int source) {
    int V = graph.size();
    vector<int> dist(V, INF);
    vector<bool> visited(V, false);
    dist[source] = 0;

    // Min distance priority queue
    using P = pair<int, int>; // pair -> {distance, vertex}
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
            int w = e.weight;
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
optional<std::vector<int>> BellmanFord_Algorithm(const AdjList<Edge>& graph, int source) {
    int V = graph.size();
    vector<int> dist(V, INF);
    dist[source] = 0;

    // Create an edge list
    vector<tuple<int,int,int>> edges;
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
            cerr << "Graph contains a negative-weight cycle!\n";
            break;
        }
    }

    return dist;
}

optional<AdjMatrix<int>> JohnsonAlgorithm(AdjList<Edge>& graph, const bool display_progress = false) {
    int V = graph.size();

    int rank, nproc;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

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
    optional<vector<int>> bellman = BellmanFord_Algorithm(extendedGraph, V);

    if (!bellman)
    {
        return std::nullopt; // BellmanFord hit a negative cycle
    }

    auto& h = *bellman;

    MPI_Bcast(h.data(), V + 1, MPI_INT, 0, MPI_COMM_WORLD);

    auto stepTwoEnd = chrono::high_resolution_clock::now();
    auto stepThreeStart = chrono::high_resolution_clock::now();

    // Step 3: reweight all edges
    // This step gets rid of all negative weights by offsetting by h(v)
    AdjList<Edge> reweightedGraph(V);

    for (int u = 0; u < V; u++)
    {
        for (const Edge& e : graph[u])
        {
            int newWeight = e.weight + h[u] - h[e.toVertex];
            reweightedGraph[u].push_back({e.toVertex, newWeight});
        }
    }

    auto stepThreeEnd = chrono::high_resolution_clock::now();
    auto stepFourStart = chrono::high_resolution_clock::now();

    // Step 4: run Dijkstra from each vertex
    // Standard priority queue based dijkstra's implementation
    // run in a for loop across the entire graph
    int verticesCompleted = 0; // progress display variable
    AdjMatrix<int> distanceMatrix;

    int numRows = V / nproc;
    int remainder = V % nproc;
    int startRow = rank * numRows + min(rank, remainder);
    int endRow = startRow + numRows + (rank < remainder ? 1 : 0);
    int localRows = endRow - startRow;

    vector<int> localflatDistances;
    localflatDistances.reserve(localRows * V);


    // Each process computes shortest paths for its assigned rows
    for (int u = startRow; u < endRow; u++)
    {
        vector<int> dist = Dijkstra_Algorithm(reweightedGraph, u);
        for (int v = 0; v < V; v++)
        {
            if (dist[v] != INF){
                localflatDistances.push_back(dist[v] - h[u] + h[v]);
            } else {
                localflatDistances.push_back(INF);
            }
        }

        // Allow user to see progress of the program when display_progress is true
        if (display_progress && rank == 0)
        {
            verticesCompleted++;
            if (rank == 0)
                cout << "\rProgress: [" << verticesCompleted << "/" << V << "] vertices completed." << flush;
        }
    }
    if (rank == 0 && display_progress)
        cout << "\rProgress: [" << verticesCompleted << "/" << V << "] vertices completed." << endl;

    // Gather all local distances to the root process
    vector<int> recvCounts(nproc);
    vector<int> displs(nproc);

    int myDataSize = localflatDistances.size();
    MPI_Allgather(&myDataSize, 1, MPI_INT, recvCounts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    int totalLen = 0;
    for (int i = 0; i < nproc; i++) {
        displs[i] = totalLen;
        totalLen += recvCounts[i];
    }

    vector<int> allflatDistances;
    if (rank == 0) {
        allflatDistances.resize(totalLen);
    }

    MPI_Gatherv(localflatDistances.data(), myDataSize, MPI_INT,
                allflatDistances.data(), recvCounts.data(), displs.data(),
                MPI_INT, 0, MPI_COMM_WORLD);

    // Reshape the flat distance array into a 2D matrix on the root process
    if (rank == 0) {
        distanceMatrix.resize(V, vector<int>(V, INF));
        int index = 0;
        for (int i = 0; i < V; i++) {
            for (int j = 0; j < V; j++) {
                distanceMatrix[i][j] = allflatDistances[index++];
            }
        }
    }

    auto stepFourEnd = chrono::high_resolution_clock::now();

    if (rank == 0)
    {
        // Display step times
        chrono::duration<double> stepOne = stepOneEnd - stepOneStart;
        chrono::duration<double> stepTwo = stepTwoEnd - stepTwoStart;
        chrono::duration<double> stepThree = stepThreeEnd - stepThreeStart;
        chrono::duration<double> stepFour = stepFourEnd - stepFourStart;

        cout << "Add Extra Vertex Elapsed Time:     " << stepOne.count()   << " Seconds\n"
             << "BellMan-Ford Elapsed Time:         " << stepTwo.count()   << " Seconds\n"
             << "Reweight Edges Elapsed Time:       " << stepThree.count() << " Seconds\n"
             << "Dijkstra/Fix weights Elapsed Time: " << stepFour.count()  << " Seconds" << endl;
    }

    // return an adjacencyMatrix for all distances
    return distanceMatrix;
}

// Function to print the results or export them to a file
void printResults(ostream& output, const AdjMatrix<int>& graph) {

    tuple<int, int, double, int> stats = getStats(graph);
    long graphSize = graph.size() * graph[0].size();

    output << endl;
    output << "Longest Distance: " << get<0>(stats) << endl;
    output << "Shortest Non-Zero Distance: " << get<1>(stats) << endl;
    output << "Average Distance: " << get<2>(stats) << endl;
    output << "INF Distance count: " << get<3>(stats) << '/' << graphSize << endl;
}

// Function to compute statistics about the shortest path distances
tuple<int,int,double,int> getStats(const AdjMatrix<int>& graph)
{
    int rows = graph.size();
    int cols = graph[0].size();
    long graphTotalSize = (long)rows * cols;

    // set max and min to extremes
    int maxVal = std::numeric_limits<int>::min();
    int minNonZero = std::numeric_limits<int>::max();
    long long total = 0;
    int numValidDistances = 0;
    int numINF = 0;


    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            int cur = graph[i][j];
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

    double average = (numValidDistances > 0) ? (double)total / numValidDistances : 0.0;

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
    int from, to, numEdges, weight;

    infile >> from >> to >> numEdges;
    cout << "Reading " << from << " x " << to << " graph with " << numEdges << " edges." << endl;

    // Initialize the adjacency list graph
    graph.assign(from, list<Edge>());

    // Read edges and populate the graph
    while (infile >> from >> to >> weight)
    {
        graph[from].push_back(Edge(to, weight));
    }
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

// Function to broadcast the graph from root process to all other processes
void broadcastGraph(AdjList<Edge>& graph, int rank) {
    int V = 0;
    std::vector<int> flat_data;

    if (rank == 0) {
        V = graph.size();

        for (int u = 0; u < V; u++) {
            for (const Edge& e : graph[u]) {
                flat_data.push_back(u);
                flat_data.push_back(e.toVertex);
                flat_data.push_back(e.weight);
            }
        }
    }

    MPI_Bcast(&V, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        graph.assign(V, std::list<Edge>());
    }

    int flat_size = flat_data.size();
    MPI_Bcast(&flat_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        flat_data.resize(flat_size);
    }

    MPI_Bcast(flat_data.data(), flat_size, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        for (int i = 0; i < flat_size; i += 3) {
            int u = flat_data[i];
            int v = flat_data[i + 1];
            int w = flat_data[i + 2];
            graph[u].push_back(Edge(v, w));
        }
    }
}

// Function to hide the cursor in the console (linux only)
void hideCursor() { cout << "\033[?25l"; }

// Function to show the cursor in the console (linux only)
void showCursor() { cout << "\033[?25h"; }


int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int N, rank, nproc;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    if (argc < 2)
    {
        if(rank == 0)
            cerr << "Usage: mpiexec -np <nproc> " << argv[0] << " <input_file> [1|0 for displaying progress]\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    ifstream infile(argv[1]);

    bool display_progress = false;
    // Optional argument to display progress since it slows down execution
    if (argc > 2)
    {
        display_progress = stoi(argv[2]) != 0;
    }

    if (!infile)
    {
        if(rank == 0)
            cerr << "Error opening file: " << argv[1] << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    // Define the graph
    AdjList<Edge> graph;

    hideCursor();

    // Read the graph from the input file
    if(rank == 0){
        readGraph(infile, graph, display_progress);
    }

    // Execute Johnson's Algorithm
    MPI_Barrier(MPI_COMM_WORLD);
    broadcastGraph(graph, rank);
    double start_time = MPI_Wtime();
    optional<AdjMatrix<int>> all_distances_opt = JohnsonAlgorithm(graph, display_progress);

    if (!all_distances_opt)
    {
        cerr << "Graph contains a negative-weight cycle!\n";
        showCursor();
        return 1;
    }

    const auto& all_distances = *all_distances_opt;

    double end_time = MPI_Wtime();

    if(rank == 0){
        showCursor();
        double elapsed_time = end_time - start_time;
        cout << "Total Elapsed time: " << elapsed_time << " seconds\n";

        // Print or export results
        printResults(cout, all_distances);
    }
    MPI_Finalize();
    return 0;
}
