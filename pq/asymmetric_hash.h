/**
* @author: yilin guo.
* @email: guoyilin1987@gmail.com
* the header file of product quantization. 
* see reference papers: product quantization for nearest neighbor search(https://hal.inria.fr/inria-00514462/document)
*/

#ifndef PQINDEXER_H__
#define PQINDEXER_H__
#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python/module.hpp>
#include <iostream>
#include <vector>
using namespace boost::python;
using namespace std;
class PQIndexer {
private:
	int k;
	int cluster_num;
	int dimension;
 	vector<vector<vector<double> > > centroids;
        vector<vector<int> > hash_table;	 
        vector<vector<vector<double> > > LoadCentroids(const string& file_prefix, int cluster_num);      
        vector<vector<double> > GetFeature(const string& filename);
        double EulerDistance(const vector<double> &fea1, const vector<double> &fea2, int ind);
        vector<vector<int> > VectorQuantization(vector<vector<vector<double> > > &centroids, vector<vector<double> > &feature);
        vector<vector<double> > DistToEveryCentroid(const vector<double> &feature);
	vector<int> GetCandidate(vector<vector<double> > &dist_table, int K);
public:
	PQIndexer(int _k, int _cluster_num, int _dimension);
        vector<int>  query(boost::python::list& query_value, int topk);	
	void createIndex(const string& featurePath, const string& centroidsPath);
};

BOOST_PYTHON_MODULE(pq)
{
         class_<PQIndexer>("PQIndexer", init<int, int, int>())
                .def("createIndex", &PQIndexer::createIndex)
                .def("query", &PQIndexer::query)
        ;
        class_<std::vector<int> >("intVec")
                .def(vector_indexing_suite<std::vector<int> >())
        ;
       
}

#endif  /* PQINDEXER_H__ */


