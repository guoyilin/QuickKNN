/**
* @author: yilin guo. 
  @email: guoyilin1987@gmail.com
  Implementation of product quantization.
*/
#include <iostream>
#include <queue>
#include <bitset>
#include <stdlib.h>
#include <set>
#include <algorithm>
#include <vector>
#include <sstream>
#include <fstream>
#include <math.h>
#include <map>
#include "asymmetric_hash.h"

using namespace std;

PQIndexer::PQIndexer(int _k, int _cluster_num, int _dimension)
{
    k = _k;
    cluster_num = _cluster_num;
    dimension = _dimension;
}

//load the centroids for each chunk offline.
vector<vector<vector<double> > > PQIndexer::LoadCentroids(const string& file_prefix, int cluster_num) {
  vector<vector<vector<double> > > all_centroids;
  for (int i = 0; i < cluster_num; ++i) {// for each chunk
    vector<vector<double> > centroids;
    char filename[256];
    string tmp = file_prefix + "/cluster";
    sprintf(filename, "%s.%d", tmp.c_str(), i+1);//read 128 centroids from file. 
    ifstream infile(filename);
    string line;
    while (getline(infile, line)) {
      istringstream str_stream(line);
      float value;
      str_stream >> value;
      vector<double> centroid;
      while (str_stream >> value) {
        centroid.push_back((double)value);
      }
      centroids.push_back(centroid);
    }
    infile.close();
    all_centroids.push_back(centroids);
  }
  return all_centroids;
}
//read features for each image.
vector<vector<double> > PQIndexer::GetFeature(const string& filename) {
  vector<vector<double> > face_feature;
  ifstream infile(filename.c_str());
  string line;
  while (getline(infile, line)) {
    istringstream str_stream(line);
    double value;
    vector<double> feature;
    while(str_stream >> value) {
      feature.push_back(value);
    }
    face_feature.push_back(feature);
  }
  infile.close();
  return face_feature;
}
double PQIndexer::EulerDistance(const vector<double> &fea1, const vector<double> &fea2, int ind) {
  double dist = 0.0;
  double minus1 = 0.0;
  for (int i = 0; i < k; i++) {
    minus1 = (fea1[ind*k+i] -fea2[i]);
    dist += minus1*minus1;//(fea1[ind*k+i] -fea2[i]) * (fea1[ind*k+i] -fea2[i]);
  }
  return dist;
}


vector<vector<int> > PQIndexer::VectorQuantization(vector<vector<vector<double> > > &centroids, vector<vector<double> > &feature) {
  vector<vector<int> > result;
  for (int i = 0; i < feature.size(); i++) {
    vector<int> hash_code;
    for (int j = 0; j < dimension/k; ++j) {//for each chunk
      float min_dist = 1000000.0;
      int closest_centroid;
      for (int l = 0; l < centroids[j].size(); ++l) {//128
        float dist = EulerDistance(feature[i], centroids[j][l], j);//find the closed centroid.
        //cout << dist << endl;
        if (dist < min_dist) {
          min_dist = dist;
          closest_centroid = l;
        }
      }
      hash_code.push_back(closest_centroid);
    }

    result.push_back(hash_code);
  }

  return result;
}

vector<vector<double> > PQIndexer::DistToEveryCentroid(const vector<double> &feature) {
  vector<vector<double> > dist_table;
  for (int i = 0; i < centroids.size(); ++i) {//for each chunk
    vector<double> dist_chunk;
    for (int j = 0; j <centroids[i].size(); ++j) {//128
      double dist = EulerDistance(feature, centroids[i][j], i);
      dist_chunk.push_back(dist);
    }
    dist_table.push_back(dist_chunk);
  }
  return dist_table;
}

struct cmp{
  bool operator() ( pair<int, int> a, pair<int, int> b ){
    return a.first < b.first; 
  }
};
vector<int> PQIndexer::GetCandidate(vector<vector<double> > &dist_table, int K) {
  vector<int> candidate;
  priority_queue<pair<double, int>, vector<pair<double, int> >, cmp> heap_topk;
  for (int i = 0; i < hash_table.size(); ++i) {// for all data in db.
 //   if (i == face_id )
 //     continue;
    double dist = 0.0;
    int thres = 20000000.0;
    if (heap_topk.size() >= K) {
      pair<double, int> top = heap_topk.top();
      thres = top.first;
    }
    for (int j = 0; j < hash_table[i].size(); ++j) {//for each chunk
      dist += dist_table[j][hash_table[i][j]];
      if (dist > thres) {
        break;
      }
    }

    if (dist > thres) continue;
    if (heap_topk.size() < K) {
      heap_topk.push(make_pair(dist, i));//face_id && i distance.
    } else {
      heap_topk.push(make_pair(dist, i));
      heap_topk.pop();
    }
  }
  int size = heap_topk.size();
  for (int i = 0; i < K; ++i) {
    pair<double, int> top = heap_topk.top();
    candidate.push_back(top.second);
   // cout << "top " << i << " " <<  top.first <<endl;
    heap_topk.pop();
  }
  return candidate;
}
void PQIndexer::createIndex(const string& featurePath, const string& centroidsPath)
{
    vector<vector<double> > feature_info = GetFeature(featurePath);
    cout << "load features finished" << endl;
    centroids =  LoadCentroids(centroidsPath, dimension/k);
    cout << "generate centroids finished" << endl;
    hash_table = VectorQuantization(centroids, feature_info);//create hash index
    cout << "hash table built" << endl;
}

vector<int>  PQIndexer::query(boost::python::list& query_value, int topk)
{
    vector<double> query_search;
    for(int i = 0; i < len(query_value); ++i)
      query_search.push_back(boost::python::extract<double>(query_value[i]));
    
    vector<vector<double> > dist_table = DistToEveryCentroid(query_search);//look up table.
    vector<int> candidate = GetCandidate(dist_table,  topk);//get top candidate.
    // for(int i =0; i < candidate.size(); i++)
    // {
    //   cout << candidate[i] << ";";
    // }
    // cout << endl;
    return candidate;
    
}

//template<class T>
// boost::python::list PQIndexer::std_vector_to_py_list(const vector<T> & v)
// {
//     boost::python::object get_iter = boost::python::iterator<std::vector<T> >();
//     boost::python::object iter = get_iter(v);
//     boost::python::list l(iter);
//     return l;
// }
/*
int main(int argc, char **argv)
{
    int k = 8;
    int cluster_num = 256;
    int dimension = 144; 
    PQIndexer indexer(k, cluster_num, dimension);
    string featurePath = "image_feature.144";
    string centroidsPath = "cluster/";
    indexer.createIndex(featurePath, centroidsPath);
    int topk = 30;
    vector<double> query = indexer.one_query;
    for(int i =0; i < query.size(); i++)
    	cout << query[i] << ";";
    cout << endl;
    vector<int> result = indexer.query(query, topk);
//    for(int i = 0 ;i < result.size(); i++)
//    {
//	cout << result[i] << endl;
//    }
    return 0;
}
*/

