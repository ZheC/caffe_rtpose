#include "rtpose/modelDescriptor.h"
#include "rtpose/modelDescriptorFactory.h"
#include "rtpose/renderFunctions.h"

#include <unordered_map>
#include "caffe/cpm/frame.h"
#include "caffe/cpm/layers/imresize_layer.hpp"
#include "caffe/cpm/layers/nms_layer.hpp"
#include "caffe/net.hpp"
#include "caffe/util/blocking_queue.hpp"

// network copy for each gpu thread
struct NetCopy {
    caffe::Net<float> *person_net;
    std::vector<int> num_people;
    int nblob_person;
    int nms_max_peaks;
    int nms_num_parts;
    std::unique_ptr<ModelDescriptor> up_model_descriptor;
    float* canvas; // GPU memory
    float* joints; // GPU memory
};

double get_wall_time() {
    struct timeval time;
    if (gettimeofday(&time,NULL)) {
        //  Handle error
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * 1e-6;
    //return (double)time.tv_usec;
}

typedef std::unordered_map<std::string, std::string> param_map;


class RTPose {   
public:
	RTPose(int deviceId);
	void warmup(int deviceId);
	int connectLimbs(std::vector< std::vector<double>> &subset,
    				 std::vector< std::vector< std::vector<double> > > &connection,
    				 const float *heatmap_pointer,
    				 const float *peaks,
    				 int max_peaks,
    			     float *joints,	
    				 ModelDescriptor *model_descriptor,
    				 int image_width,
    				 int image_height);
	int distanceThresholdPeaks(const float *in_peaks, int max_peaks, float *peaks, 
		ModelDescriptor *model_descriptor);
	int connectLimbsCOCO(std::vector< std::vector<double>> &subset,
    					 std::vector< std::vector< std::vector<double> > > &connection,
    					 const float *heatmap_pointer,
    					 const float *in_peaks,
    					 int max_peaks,
				 	     float *joints,
    					 ModelDescriptor *model_descriptor,
    					 int image_width,
    					 int image_height);
	Frame processFrame(Frame frame, int image_width, int image_height, int part_to_show, int googly_eyes);
	void render(float *heatmaps /*GPU*/, int image_width, int image_height, int part_to_show, int googly_eyes);
	Frame postProcessFrame(Frame frame, int image_width, int image_height);

	std::vector<unsigned char> run(std::vector<unsigned char> input, param_map & params);
	Frame getFrame(cv::Mat input, int image_width, int image_height);

private:
	NetCopy netcopy;

	std::string PERSON_DETECTOR_CAFFEMODEL = "model/coco/pose_iter_440000.caffemodel"; //person detector
	std::string PERSON_DETECTOR_PROTO = "model/coco/pose_deploy_linevec.prototxt"; //person detector
	// Global parameters
	int MAX_RESOLUTION_WIDTH = 1280;
	int MAX_RESOLUTION_HEIGHT = 720;
	int NET_RESOLUTION_WIDTH = 496;
	int NET_RESOLUTION_HEIGHT = 368;
	int BATCH_SIZE = 1;      // "Number of scales to average"
	double SCALE_GAP = 0.3;  // "Scale gap between scales. No effect unless num_scales>1"
	double START_SCALE = 1;  // "Initial scale. Must cv::Match net_resolution"
	int part_to_show = 0;    // "Part to show from the start.");
	int global_counter = 0;

	const int MAX_PEOPLE = RENDER_MAX_PEOPLE;  // defined in render_functions.hpp
	const int BOX_SIZE = 368;
	const int BUFFER_SIZE = 4;    //affects latency
	const int MAX_NUM_PARTS = 70;

	// from global:
	float nms_threshold;
	int connect_min_subset_cnt;
    float connect_min_subset_score;
    float connect_inter_threshold;
    int connect_inter_min_above_threshold;

};
