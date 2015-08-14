//==========================================================================// 
// Copyright 2009 Google Inc.                                               // 
//                                                                          // 
// Licensed under the Apache License, Version 2.0 (the "License");          // 
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  // 
//                                                                          //
//      http://www.apache.org/licenses/LICENSE-2.0                          //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        // 
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. // 
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
//==========================================================================//
//
// Author: D. Sculley
// dsculley@google.com or dsculley@cs.tufts.edu   
//
// Implementation of sf-cluster-centers.h

#include <assert.h>
#include <cmath>
#include <cstdlib>
#include <float.h>
#include <iostream>
#include <fstream>
#include "sf-cluster-centers.h"

#include <algorithm>

// ---

// for sorting by distance value
/* qsort version
int element_compare_( void const *a, void  const *b ) {
	return ( (( element * ) a )->value - (( element * ) b )->value );
}

// sort desc
int element_compare( void const *a, void  const *b ) {
	element c, d;
	 c = *( element * ) a; 
	 d = *( element * ) b;
	 return ( d.value - c.value );
}
*/

int element_compare( element c, element d ) {
	//printf ("1st: %d/%f, 2nd: %d/%f\n", c.index, c.value, d.index, d.value );
	return ( c.value > d.value );
}

// ---

SfClusterCenters::SfClusterCenters(int dimensionality)
  : dimensionality_(dimensionality) {
  assert(dimensionality_ >= 0);
}

SfClusterCenters::SfClusterCenters(int dimensionality,
				   int num_clusters)
  : dimensionality_(dimensionality) {
  assert(dimensionality_ >= 0);
  assert(num_clusters >= 0);
  SfWeightVector w(dimensionality_);
  for (int i = 0; i < num_clusters; ++i) {
    cluster_centers_.push_back(w);
  } 
}

SfClusterCenters::SfClusterCenters(const string& file_name) {
  long int buffer_size = 1 * 1024 * 1024; // 1MB
  char* local_buffer = new char[buffer_size];
  std::ifstream file_stream(file_name.c_str(), std::ifstream::in);
  file_stream.rdbuf()->pubsetbuf(local_buffer, buffer_size); 
  if (!file_stream) {
    std::cerr << "Error reading file " << file_name << std::endl;
    exit(1);
  }

  string line_string;
  while (getline(file_stream, line_string)) {
    AddClusterCenter(SfWeightVector(line_string));
  }
  
  delete[] local_buffer;

}

void SfClusterCenters::AddClusterCenter(const SfWeightVector& new_center) {
  cluster_centers_.push_back(new_center);
  if (new_center.GetDimensions() > dimensionality_) {
    dimensionality_ = new_center.GetDimensions();
  }
}

void SfClusterCenters::AddClusterCenterAt(const SfSparseVector& x) {
  SfWeightVector new_center(dimensionality_);
  new_center.AddVector(x, 1.0);
  AddClusterCenter(new_center);
}

float SfClusterCenters::SqDistanceToCenterId(int center_id,
					     const SfSparseVector& x) const {
  assert(center_id >= 0 &&
	 static_cast<unsigned int>(center_id) < cluster_centers_.size());
  // ||a - b||^2 = a^2 - 2ab + b^2
  float squared_distance =
    x.GetSquaredNorm() -
    2 * cluster_centers_[center_id].InnerProduct(x, 1.0) +
    cluster_centers_[center_id].GetSquaredNorm();
  return squared_distance;
}

float SfClusterCenters::SqDistanceToClosestCenter(
      const SfSparseVector& x,
      int* closest_center_id) const {
  assert(!cluster_centers_.empty());
  assert(closest_center_id != NULL);
  float min_distance = FLT_MAX;
  int best_center = 0;
  for (unsigned int i = 0; i < cluster_centers_.size(); ++i) {
    float distance_i = SqDistanceToCenterId(i, x);
    if (distance_i < min_distance) {
      min_distance = distance_i;
      best_center = i;
    }
  }
  *closest_center_id = best_center;
  return min_distance;
}

const SfWeightVector& SfClusterCenters::ClusterCenter(int center_id) const {
  return cluster_centers_[center_id];
}

SfWeightVector* SfClusterCenters::MutableClusterCenter(int center_id) {
  return &(cluster_centers_[center_id]);
}

string SfClusterCenters::AsString() {
  string output_string;
  for (unsigned int i = 0; i < cluster_centers_.size(); ++i) {
    output_string += cluster_centers_[i].AsString();
    output_string += "\n";
  }
  return output_string;
}

SfSparseVector* SfClusterCenters::MapVectorToCenters(
    const SfSparseVector& x,
    ClusterCenterMappingType type,
    float p,
    float t,
    int s ) const {
  SfSparseVector* mapped_x = new SfSparseVector(x);
  assert(mapped_x != NULL);
  mapped_x->ClearFeatures();
  
  unsigned int num_clusters = cluster_centers_.size();
  struct element distances[num_clusters];
  int element_ranks[num_clusters];
  float element_distances[num_clusters];
  // num_to_select is s

  for (unsigned int i = 1; i <= num_clusters; ++i) {
    float d = SqDistanceToCenterId(i - 1, x);
    switch (type) {
    case SQUARED_DISTANCE:
      break;
    case RBF_KERNEL:
      d = exp(-1.0 * p * d);
      if ( d < t ) {
      	d = 0.0;
      }
      break;
    default:
      std::cerr << "ClusterCenterMappingType " << type << "not supported.";
      exit(1);
    }
    if ( s > 0 and type != SQUARED_DISTANCE ) {
	    distances[i - 1].index = i;
	    distances[i - 1].value = d;
	    element_distances[i - 1] = d;
	} else {
		mapped_x->PushPair(i, d);
	}
  }
  
  	// select s closest clusters
  	if ( s > 0 and type != SQUARED_DISTANCE ) {
  
		for ( unsigned int n = 0; n < 10; n++ ) {
			//printf ("index: %d, value: %f\n", distances[n].index, distances[n].value );
		}  
		//printf ("\n\n");

		//qsort( distances, num_clusters, sizeof( element ), element_compare );
		std::sort( distances, distances + num_clusters, &element_compare );

		for ( unsigned int n = 0; n < 10; n++ ) {
			//printf ("index: %d, value: %f\n", distances[n].index, distances[n].value );
		}  
		//exit( 0 );

		// compute ranks 0...n-1  
		for ( unsigned int n = 0; n < num_clusters; n++ ) {
			int index = distances[n].index;
			float value = distances[n].value;
			element_ranks[index - 1] = n;
			//printf( "%d / %f - rank %d\n", index, value, n );
		}  
		//exit( 0 );

		for ( unsigned int n = 0; n < num_clusters; n++ ) {
			if ( element_ranks[n] < s ) {
				mapped_x->PushPair( n + 1, element_distances[n] );
				//printf( "index: %d\trank:%d\tdistance: %f \n", n+1, element_ranks[n], element_distances[n] );

			} 
		}  
		//exit( 0 );
	}
  
  return mapped_x;
}

