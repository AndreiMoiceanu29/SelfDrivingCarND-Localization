/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  std::default_random_engine generator;
  std::normal_distribution<double> x_dist(x,std[0]);
  std::normal_distribution<double> y_dist(y,std[1]);
  std::normal_distribution<double> theta_dist(theta,std[2]);
  num_particles = 50;  // TODO: Set the number of particles
  
  
  for(int i = 0; i<num_particles; i++){
    Particle current_particle;
    current_particle.id = i;
    current_particle.x = x_dist(generator);
    current_particle.y = y_dist(generator);
    current_particle.theta = theta_dist(generator);
    current_particle.weight = 1.0;
    weights.push_back(current_particle.weight);
    particles.push_back(current_particle);
  } 
  is_initialized = true;
}

void ParticleFilter::prediction(double dt, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  std::default_random_engine generator;
  double x, y,theta;
  
  for(int i = 0; i<num_particles;i++){
    if(fabs(yaw_rate) > 0.0001){
    //Predict the future states.
    x = particles[i].x + (velocity/yaw_rate*(sin(particles[i].theta + yaw_rate * dt) - sin(particles[i].theta)));
    y = particles[i].y + (velocity/yaw_rate*(cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * dt)));
    theta = particles[i].theta +  yaw_rate * dt;
    } else{
      x = particles[i].x + velocity * cos(particles[i].theta) * dt;
      y = particles[i].y + velocity * sin(particles[i].theta) * dt;
      theta = particles[i].theta;
    }
    std::normal_distribution<double> x_dist(x, std_pos[0]);
    std::normal_distribution<double> y_dist(y, std_pos[1]);
    std::normal_distribution<double> theta_dist(theta,std_pos[2]);
    //Create normal distributions with mean around the predicted states, and std_dev std_pos.
    
    //Update the particle states.
    particles[i].x = x_dist(generator);
    particles[i].y = y_dist(generator);
    particles[i].theta = theta_dist(generator);
  }

}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations,double sensor_range) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  double distance, current_dist;
  int idx;
  for(int i = 0; i<observations.size();i++){
    idx = -1;
    distance = sensor_range * sqrt(2);
    for(int k = 0; k<predicted.size();k++){
      current_dist = dist(predicted[k].x,predicted[k].y,observations[i].x,observations[i].y);
      if(current_dist < distance){
        distance = current_dist;
  		idx = predicted[k].id;
      }
    }
    observations[i].id = idx;
  }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> observations, 
                                   const Map map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  double weights_sum = 0.0;
  
  for(int i = 0; i<num_particles;i++){
    vector<LandmarkObs> tr_observations;
    //Apply the transformation from car coordinates to map coords.
   for(int k = 0 ; k<observations.size();k++){
     LandmarkObs obs;
     obs.id = k;
     obs.x = observations[k].x * cos(particles[i].theta) - observations[k].y * sin(particles[i].theta) + particles[i].x;
     obs.y = observations[k].y * cos(particles[i].theta) + observations[k].x * sin(particles[i].theta) + particles[i].y;
     tr_observations.push_back(obs);
   }
    vector<LandmarkObs> keept_observations;
    
    //Removing the observations that are not in range of the sensors, for optimizations pruporse.
    for(int j = 0; j< map_landmarks.landmark_list.size();j++){
      LandmarkObs obsv;
      Map::single_landmark_s curr_landmark = map_landmarks.landmark_list[j];
      if((fabs(curr_landmark.x_f - particles[i].x) <= sensor_range) && (fabs(curr_landmark.y_f - particles[i].y) <= sensor_range)){
        obsv.x = curr_landmark.x_f;
        obsv.y = curr_landmark.y_f;
        obsv.id = curr_landmark.id_i;
        keept_observations.push_back(obsv);
      }
    }
    dataAssociation(keept_observations,tr_observations, sensor_range);
    particles[i].weight = 1.0;
    //Update the weights using the multivariate gaussian.
    double sigma_x = std_landmark[0];
    double sigma_y = std_landmark[1];
    double normalizer = (1.0/(2.0 * M_PI * sigma_x * sigma_y));
    
    for(int j = 0 ; j<tr_observations.size();j++){
      double multiv_gaussian = 1.0;
      double tr_x = tr_observations[j].x;
      double tr_y = tr_observations[j].y;
      int tr_id = tr_observations[j].id;
      for(int k = 0; k < keept_observations.size(); k++){
        int k_id = keept_observations[k].id;
        double k_x = keept_observations[k].x;
        double k_y = keept_observations[k].y;
        if(tr_id == k_id){
          multiv_gaussian = normalizer * exp(-1.0 * (pow(k_x-tr_x,2)/(2.0*pow(sigma_x,2)) + pow(k_y-tr_y,2)/(2.0*pow(sigma_y,2))));
          particles[i].weight *= multiv_gaussian;
        }
      }
      
    }
    weights_sum += particles[i].weight;
    
  }
  //Normalize the weights
  for(int i = 0; i<num_particles;i++){
    particles[i].weight /= weights_sum;
    weights[i] = particles[i].weight;
  }

}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  vector<Particle> resampled_particles;
  std::default_random_engine generator;
  std::uniform_int_distribution<int> index_distributions(0,num_particles - 1);
  int idx = index_distributions(generator);
  double beta = 0.0;
  double max_weight = *std::max_element(weights.begin(),weights.end());
  for(int i=0; i<num_particles; i++){
    std::uniform_real_distribution<double> weight_dist(0,2.0 * max_weight);
    beta += weight_dist(generator);
    
    while (weights[idx] < beta){
      beta -= weights[idx];
      idx = (idx + 1) % num_particles;
    }
    resampled_particles.push_back(particles[idx]);
    
  }
  particles = resampled_particles;

}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}