#include <algorithm>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <iostream>
#include <mutex>
#include <string>
#include <vector>
#include <utility> //std::pair

#include <pthread.h>
#include <time.h>
#include <signal.h>
#include <stdio.h>  // snprintf
#include <unistd.h>
#include <stdlib.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/types.h>

#include <boost/thread/thread.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <google/protobuf/text_format.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


#include "rtpose/rtpose_demo.hpp"

struct ColumnCompare
{
    bool operator()(const std::vector<double>& lhs,
                    const std::vector<double>& rhs) const
    {
        return lhs[2] > rhs[2];
        //return lhs[0] > rhs[0];
    }
};


void process_and_pad_image(float* target, cv::Mat oriImg, int tw, int th, bool normalize) {
    int ow = oriImg.cols;
    int oh = oriImg.rows;
    int offset2_target = tw * th;

    int padw = (tw-ow)/2;
    int padh = (th-oh)/2;
    //LOG(ERROR) << " padw " << padw << " padh " << padh;
    CHECK_GE(padw,0) << "Image too big for target size.";
    CHECK_GE(padh,0) << "Image too big for target size.";
    //parallel here
    unsigned char* pointer = (unsigned char*)(oriImg.data);

    for(int c = 0; c < 3; c++) {
        for(int y = 0; y < th; y++) {
            int oy = y - padh;
            for(int x = 0; x < tw; x++) {
                int ox = x - padw;
                if (ox>=0 && ox < ow && oy>=0 && oy < oh ) {
                    if (normalize)
                        target[c * offset2_target + y * tw + x] = float(pointer[(oy * ow + ox) * 3 + c])/256.0f - 0.5f;
                    else
                        target[c * offset2_target + y * tw + x] = float(pointer[(oy * ow + ox) * 3 + c]);
                }
                else {
                    target[c * offset2_target + y * tw + x] = 0;
                }
            }
        }
    }
}

int RTPose::connectLimbs(
    std::vector< std::vector<double>> &subset,
    std::vector< std::vector< std::vector<double> > > &connection,
    const float *heatmap_pointer,
    const float *peaks,
    int max_peaks,
    float *joints,
    ModelDescriptor *model_descriptor,
    int image_width,
    int image_height) {

        const auto num_parts = model_descriptor->get_number_parts();
        const auto limbSeq = model_descriptor->get_limb_sequence();
        const auto mapIdx = model_descriptor->get_map_idx();
        const auto number_limb_seq = model_descriptor->number_limb_sequence();

        int SUBSET_CNT = num_parts+2;
        int SUBSET_SCORE = num_parts+1;
        int SUBSET_SIZE = num_parts+3;

        CHECK_EQ(num_parts, 15);
        CHECK_EQ(number_limb_seq, 14);

        int peaks_offset = 3*(max_peaks+1);
        subset.clear();
        connection.clear();

        for(int k = 0; k < number_limb_seq; k++) {
            const float* map_x = heatmap_pointer + mapIdx[2*k] * NET_RESOLUTION_HEIGHT * NET_RESOLUTION_WIDTH;
            const float* map_y = heatmap_pointer + mapIdx[2*k+1] * NET_RESOLUTION_HEIGHT * NET_RESOLUTION_WIDTH;

            const float* candA = peaks + limbSeq[2*k]*peaks_offset;
            const float* candB = peaks + limbSeq[2*k+1]*peaks_offset;

            std::vector< std::vector<double> > connection_k;
            int nA = candA[0];
            int nB = candB[0];

            // add parts into the subset in special case
            if (nA ==0 && nB ==0) {
                continue;
            }
            else if (nA ==0) {
                for(int i = 1; i <= nB; i++) {
                    std::vector<double> row_vec(SUBSET_SIZE, 0);
                    row_vec[ limbSeq[2*k+1] ] = limbSeq[2*k+1]*peaks_offset + i*3 + 2; //store the index
                    row_vec[SUBSET_CNT] = 1; //last number in each row is the parts number of that person
                    row_vec[SUBSET_SCORE] = candB[i*3+2]; //second last number in each row is the total score
                    subset.push_back(row_vec);
                }
                continue;
            }
            else if (nB ==0) {
                for(int i = 1; i <= nA; i++) {
                    std::vector<double> row_vec(SUBSET_SIZE, 0);
                    row_vec[ limbSeq[2*k] ] = limbSeq[2*k]*peaks_offset + i*3 + 2; //store the index
                    row_vec[SUBSET_CNT] = 1; //last number in each row is the parts number of that person
                    row_vec[SUBSET_SCORE] = candA[i*3+2]; //second last number in each row is the total score
                    subset.push_back(row_vec);
                }
                continue;
            }

            std::vector< std::vector<double>> temp;
            const int num_inter = 10;

            for(int i = 1; i <= nA; i++) {
                for(int j = 1; j <= nB; j++) {
                    float s_x = candA[i*3];
                    float s_y = candA[i*3+1];
                    float d_x = candB[j*3] - candA[i*3];
                    float d_y = candB[j*3+1] - candA[i*3+1];
                    float norm_vec = sqrt( pow(d_x,2) + pow(d_y,2) );
                    if (norm_vec<1e-6) {
                        continue;
                    }
                    float vec_x = d_x/norm_vec;
                    float vec_y = d_y/norm_vec;

                    float sum = 0;
                    int count = 0;

                    for(int lm=0; lm < num_inter; lm++) {
                        int my = round(s_y + lm*d_y/num_inter);
                        int mx = round(s_x + lm*d_x/num_inter);
                        int idx = my * NET_RESOLUTION_WIDTH + mx;
                        float score = (vec_x*map_x[idx] + vec_y*map_y[idx]);
                        if (score > connect_inter_threshold) {
                            sum = sum + score;
                            count ++;
                        }
                    }
                    //float score = sum / count; // + std::min((130/dist-1),0.f)

                    if (count > connect_inter_min_above_threshold) {//num_inter*0.8) { //thre/2
                        // parts score + cpnnection score
                        std::vector<double> row_vec(4, 0);
                        row_vec[3] = sum/count + candA[i*3+2] + candB[j*3+2]; //score_all
                        row_vec[2] = sum/count;
                        row_vec[0] = i;
                        row_vec[1] = j;
                        temp.push_back(row_vec);
                    }
                }
            }

        //** select the top num connection, assuming that each part occur only once
        // sort rows in descending order based on parts + connection score
        if (temp.size() > 0)
            std::sort(temp.begin(), temp.end(), ColumnCompare());

        int num = std::min(nA, nB);
        int cnt = 0;
        std::vector<int> occurA(nA, 0);
        std::vector<int> occurB(nB, 0);

        for(int row =0; row < temp.size(); row++) {
            if (cnt==num) {
                break;
            }
            else{
                int i = int(temp[row][0]);
                int j = int(temp[row][1]);
                float score = temp[row][2];
                if ( occurA[i-1] == 0 && occurB[j-1] == 0 ) { // && score> (1+thre)
                    std::vector<double> row_vec(3, 0);
                    row_vec[0] = limbSeq[2*k]*peaks_offset + i*3 + 2;
                    row_vec[1] = limbSeq[2*k+1]*peaks_offset + j*3 + 2;
                    row_vec[2] = score;
                    connection_k.push_back(row_vec);
                    cnt = cnt+1;
                    occurA[i-1] = 1;
                    occurB[j-1] = 1;
                }
            }
        }

        if (k==0) {
            std::vector<double> row_vec(num_parts+3, 0);
            for(int i = 0; i < connection_k.size(); i++) {
                double indexA = connection_k[i][0];
                double indexB = connection_k[i][1];
                row_vec[limbSeq[0]] = indexA;
                row_vec[limbSeq[1]] = indexB;
                row_vec[SUBSET_CNT] = 2;
                // add the score of parts and the connection
                row_vec[SUBSET_SCORE] = peaks[int(indexA)] + peaks[int(indexB)] + connection_k[i][2];
                subset.push_back(row_vec);
            }
        }
        else{
            if (connection_k.size()==0) {
                continue;
            }
            // A is already in the subset, find its connection B
            for(int i = 0; i < connection_k.size(); i++) {
                int num = 0;
                double indexA = connection_k[i][0];
                double indexB = connection_k[i][1];

                for(int j = 0; j < subset.size(); j++) {
                    if (subset[j][limbSeq[2*k]] == indexA) {
                        subset[j][limbSeq[2*k+1]] = indexB;
                        num = num+1;
                        subset[j][SUBSET_CNT] = subset[j][SUBSET_CNT] + 1;
                        subset[j][SUBSET_SCORE] = subset[j][SUBSET_SCORE] + peaks[int(indexB)] + connection_k[i][2];
                    }
                }
                // if can not find partA in the subset, create a new subset
                if (num==0) {
                    std::vector<double> row_vec(SUBSET_SIZE, 0);
                    row_vec[limbSeq[2*k]] = indexA;
                    row_vec[limbSeq[2*k+1]] = indexB;
                    row_vec[SUBSET_CNT] = 2;
                    row_vec[SUBSET_SCORE] = peaks[int(indexA)] + peaks[int(indexB)] + connection_k[i][2];
                    subset.push_back(row_vec);
                }
            }
        }
    }

    //** joints by deleting some rows of subset which has few parts occur
    int cnt = 0;
    for(int i = 0; i < subset.size(); i++) {
        if (subset[i][SUBSET_CNT]>=connect_min_subset_cnt && (subset[i][SUBSET_SCORE]/subset[i][SUBSET_CNT])>connect_min_subset_score) {
            for(int j = 0; j < num_parts; j++) {
                int idx = int(subset[i][j]);
                if (idx) {
                    joints[cnt*num_parts*3 + j*3 +2] = peaks[idx];
                    joints[cnt*num_parts*3 + j*3 +1] = peaks[idx-1] * image_height/ (float)NET_RESOLUTION_HEIGHT;
                    joints[cnt*num_parts*3 + j*3] = peaks[idx-2] * image_width/ (float)NET_RESOLUTION_WIDTH;
                }
                else{
                    joints[cnt*num_parts*3 + j*3 +2] = 0;
                    joints[cnt*num_parts*3 + j*3 +1] = 0;
                    joints[cnt*num_parts*3 + j*3] = 0;
                }
            }
            cnt++;
            if (cnt==MAX_PEOPLE) break;
        }
    }

    return cnt;
}

int RTPose::distanceThresholdPeaks(const float *in_peaks, int max_peaks,
    float *peaks, ModelDescriptor *model_descriptor) {
    // Post-process peaks to remove those which are within sqrt(dist_threshold2)
    // of each other.

    const auto num_parts = model_descriptor->get_number_parts();
    const float dist_threshold2 = 6*6;
    int peaks_offset = 3*(max_peaks+1);

    int total_peaks = 0;
    for(int p = 0; p < num_parts; p++) {
        const float *pipeaks = in_peaks + p*peaks_offset;
        float *popeaks = peaks + p*peaks_offset;
        int num_in_peaks = int(pipeaks[0]);
        int num_out_peaks = 0; // Actual number of peak count
        for (int c1=0;c1<num_in_peaks;c1++) {
            float x1 = pipeaks[(c1+1)*3+0];
            float y1 = pipeaks[(c1+1)*3+1];
            float s1 = pipeaks[(c1+1)*3+2];
            bool keep = true;
            for (int c2=0;c2<num_out_peaks;c2++) {
                float x2 = popeaks[(c2+1)*3+0];
                float y2 = popeaks[(c2+1)*3+1];
                float s2 = popeaks[(c2+1)*3+2];
                float dist2 = (x1-x2)*(x1-x2) + (y1-y2)*(y1-y2);
                if (dist2<dist_threshold2) {
                    // This peak is too close to a peak already in the output buffer
                    // so don't add it.
                    keep = false;
                    if (s1>s2) {
                        // It's better than the one in the output buffer
                        // so we swap it.
                        popeaks[(c2+1)*3+0] = x1;
                        popeaks[(c2+1)*3+1] = y1;
                        popeaks[(c2+1)*3+2] = s1;
                    }
                }
            }
            if (keep && num_out_peaks<max_peaks) {
                // We don't already have a better peak within the threshold distance
                popeaks[(num_out_peaks+1)*3+0] = x1;
                popeaks[(num_out_peaks+1)*3+1] = y1;
                popeaks[(num_out_peaks+1)*3+2] = s1;
                num_out_peaks++;
            }
        }
        // if (num_in_peaks!=num_out_peaks) {
            //LOG(INFO) << "Part: " << p << " in peaks: "<< num_in_peaks << " out: " << num_out_peaks;
        // }
        popeaks[0] = float(num_out_peaks);
        total_peaks += num_out_peaks;
    }
    return total_peaks;
}

int RTPose::connectLimbsCOCO(
    std::vector< std::vector<double>> &subset,
    std::vector< std::vector< std::vector<double> > > &connection,
    const float *heatmap_pointer,
    const float *in_peaks,
    int max_peaks,
    float *joints,
    ModelDescriptor *model_descriptor,
    int image_width,
    int image_height) {
        /* Parts Connection ---------------------------------------*/
        const auto num_parts = model_descriptor->get_number_parts();
        const auto limbSeq = model_descriptor->get_limb_sequence();
        const auto mapIdx = model_descriptor->get_map_idx();
        const auto number_limb_seq = model_descriptor->number_limb_sequence();

        CHECK_EQ(num_parts, 18) << "Wrong connection function for model";
        CHECK_EQ(number_limb_seq, 19) << "Wrong connection function for model";

        int SUBSET_CNT = num_parts+2;
        int SUBSET_SCORE = num_parts+1;
        int SUBSET_SIZE = num_parts+3;

        const int peaks_offset = 3*(max_peaks+1);

        const float *peaks = in_peaks;
        subset.clear();
        connection.clear();

        for(int k = 0; k < number_limb_seq; k++) {
            const float* map_x = heatmap_pointer + mapIdx[2*k] * NET_RESOLUTION_HEIGHT * NET_RESOLUTION_WIDTH;
            const float* map_y = heatmap_pointer + mapIdx[2*k+1] * NET_RESOLUTION_HEIGHT * NET_RESOLUTION_WIDTH;

            const float* candA = peaks + limbSeq[2*k]*peaks_offset;
            const float* candB = peaks + limbSeq[2*k+1]*peaks_offset;

            std::vector< std::vector<double> > connection_k;
            int nA = candA[0];
            int nB = candB[0];

            // add parts into the subset in special case
            if (nA ==0 && nB ==0) {
                continue;
            } else if (nA ==0) {
                for(int i = 1; i <= nB; i++) {
                    int num = 0;
                    int indexB = limbSeq[2*k+1];
                    for(int j = 0; j < subset.size(); j++) {
                            int off = limbSeq[2*k+1]*peaks_offset + i*3 + 2;
                            if (subset[j][indexB] == off) {
                                    num = num+1;
                                    continue;
                            }
                    }
                    if (num!=0) {
                        //LOG(INFO) << " else if (nA==0) shouldn't have any nB already assigned?";
                    } else {
                        std::vector<double> row_vec(SUBSET_SIZE, 0);
                        row_vec[ limbSeq[2*k+1] ] = limbSeq[2*k+1]*peaks_offset + i*3 + 2; //store the index
                        row_vec[SUBSET_CNT] = 1; //last number in each row is the parts number of that person
                        row_vec[SUBSET_SCORE] = candB[i*3+2]; //second last number in each row is the total score
                        subset.push_back(row_vec);
                    }
                    //LOG(INFO) << "nA==0 New subset on part " << k << " subsets: " << subset.size();
                }
                continue;
            } else if (nB ==0) {
                for(int i = 1; i <= nA; i++) {
                    int num = 0;
                    int indexA = limbSeq[2*k];
                    for(int j = 0; j < subset.size(); j++) {
                            int off = limbSeq[2*k]*peaks_offset + i*3 + 2;
                            if (subset[j][indexA] == off) {
                                    num = num+1;
                                    continue;
                            }
                    }
                    if (num==0) {
                        std::vector<double> row_vec(SUBSET_SIZE, 0);
                        row_vec[ limbSeq[2*k] ] = limbSeq[2*k]*peaks_offset + i*3 + 2; //store the index
                        row_vec[SUBSET_CNT] = 1; //last number in each row is the parts number of that person
                        row_vec[SUBSET_SCORE] = candA[i*3+2]; //second last number in each row is the total score
                        subset.push_back(row_vec);
                        //LOG(INFO) << "nB==0 New subset on part " << k << " subsets: " << subset.size();
                    } else {
                        //LOG(INFO) << "nB==0 discarded would have added";
                    }
                }
                continue;
            }

            std::vector< std::vector<double>> temp;
            const int num_inter = 10;

            for(int i = 1; i <= nA; i++) {
                for(int j = 1; j <= nB; j++) {
                    float s_x = candA[i*3];
                    float s_y = candA[i*3+1];
                    float d_x = candB[j*3] - candA[i*3];
                    float d_y = candB[j*3+1] - candA[i*3+1];
                    float norm_vec = sqrt( d_x*d_x + d_y*d_y );
                    if (norm_vec<1e-6) {
                        // The peaks are coincident. Don't connect them.
                        continue;
                    }
                    float vec_x = d_x/norm_vec;
                    float vec_y = d_y/norm_vec;

                    float sum = 0;
                    int count = 0;

                    for(int lm=0; lm < num_inter; lm++) {
                        int my = round(s_y + lm*d_y/num_inter);
                        int mx = round(s_x + lm*d_x/num_inter);
                        if (mx>=NET_RESOLUTION_WIDTH) {
                            //LOG(ERROR) << "mx " << mx << "out of range";
                            mx = NET_RESOLUTION_WIDTH-1;
                        }
                        if (my>=NET_RESOLUTION_HEIGHT) {
                            //LOG(ERROR) << "my " << my << "out of range";
                            my = NET_RESOLUTION_HEIGHT-1;
                        }
                        CHECK_GE(mx,0);
                        CHECK_GE(my,0);
                        int idx = my * NET_RESOLUTION_WIDTH + mx;
                        float score = (vec_x*map_x[idx] + vec_y*map_y[idx]);
                        if (score > connect_inter_threshold) {
                            sum = sum + score;
                            count ++;
                        }
                    }
                    //float score = sum / count; // + std::min((130/dist-1),0.f)

                    if (count > connect_inter_min_above_threshold) {//num_inter*0.8) { //thre/2
                        // parts score + cpnnection score
                        std::vector<double> row_vec(4, 0);
                        row_vec[3] = sum/count + candA[i*3+2] + candB[j*3+2]; //score_all
                        row_vec[2] = sum/count;
                        row_vec[0] = i;
                        row_vec[1] = j;
                        temp.push_back(row_vec);
                    }
                }
            }

            //** select the top num connection, assuming that each part occur only once
            // sort rows in descending order based on parts + connection score
            if (temp.size() > 0)
                std::sort(temp.begin(), temp.end(), ColumnCompare());

            int num = std::min(nA, nB);
            int cnt = 0;
            std::vector<int> occurA(nA, 0);
            std::vector<int> occurB(nB, 0);

            for(int row =0; row < temp.size(); row++) {
                if (cnt==num) {
                    break;
                }
                else{
                    int i = int(temp[row][0]);
                    int j = int(temp[row][1]);
                    float score = temp[row][2];
                    if ( occurA[i-1] == 0 && occurB[j-1] == 0 ) { // && score> (1+thre)
                        std::vector<double> row_vec(3, 0);
                        row_vec[0] = limbSeq[2*k]*peaks_offset + i*3 + 2;
                        row_vec[1] = limbSeq[2*k+1]*peaks_offset + j*3 + 2;
                        row_vec[2] = score;
                        connection_k.push_back(row_vec);
                        cnt = cnt+1;
                        occurA[i-1] = 1;
                        occurB[j-1] = 1;
                    }
                }
            }

            //** cluster all the joints candidates into subset based on the part connection
            // initialize first body part connection 15&16
            if (k==0) {
                std::vector<double> row_vec(num_parts+3, 0);
                for(int i = 0; i < connection_k.size(); i++) {
                    double indexB = connection_k[i][1];
                    double indexA = connection_k[i][0];
                    row_vec[limbSeq[0]] = indexA;
                    row_vec[limbSeq[1]] = indexB;
                    row_vec[SUBSET_CNT] = 2;
                    // add the score of parts and the connection
                    row_vec[SUBSET_SCORE] = peaks[int(indexA)] + peaks[int(indexB)] + connection_k[i][2];
                    //LOG(INFO) << "New subset on part " << k << " subsets: " << subset.size();
                    subset.push_back(row_vec);
                }
            }/* else if (k==17 || k==18) { // TODO: Check k numbers?
                //   %add 15 16 connection
                for(int i = 0; i < connection_k.size(); i++) {
                    double indexA = connection_k[i][0];
                    double indexB = connection_k[i][1];

                    for(int j = 0; j < subset.size(); j++) {
                    // if subset(j, indexA) == partA(i) && subset(j, indexB) == 0
                    //         subset(j, indexB) = partB(i);
                    // elseif subset(j, indexB) == partB(i) && subset(j, indexA) == 0
                    //         subset(j, indexA) = partA(i);
                    // end
                        if (subset[j][limbSeq[2*k]] == indexA && subset[j][limbSeq[2*k+1]]==0) {
                            subset[j][limbSeq[2*k+1]] = indexB;
                        } else if (subset[j][limbSeq[2*k+1]] == indexB && subset[j][limbSeq[2*k]]==0) {
                            subset[j][limbSeq[2*k]] = indexA;
                        }
                }
                continue;
            }
        }*/ else{
            if (connection_k.size()==0) {
                continue;
            }

            // A is already in the subset, find its connection B
            for(int i = 0; i < connection_k.size(); i++) {
                int num = 0;
                double indexA = connection_k[i][0];
                double indexB = connection_k[i][1];

                for(int j = 0; j < subset.size(); j++) {
                    if (subset[j][limbSeq[2*k]] == indexA) {
                        subset[j][limbSeq[2*k+1]] = indexB;
                        num = num+1;
                        subset[j][SUBSET_CNT] = subset[j][SUBSET_CNT] + 1;
                        subset[j][SUBSET_SCORE] = subset[j][SUBSET_SCORE] + peaks[int(indexB)] + connection_k[i][2];
                    }
                }
                // if can not find partA in the subset, create a new subset
                if (num==0) {
                    //LOG(INFO) << "New subset on part " << k << " subsets: " << subset.size();
                    std::vector<double> row_vec(SUBSET_SIZE, 0);
                    row_vec[limbSeq[2*k]] = indexA;
                    row_vec[limbSeq[2*k+1]] = indexB;
                    row_vec[SUBSET_CNT] = 2;
                    row_vec[SUBSET_SCORE] = peaks[int(indexA)] + peaks[int(indexB)] + connection_k[i][2];
                    subset.push_back(row_vec);
                }
            }
        }
    }

    //** joints by deleteing some rows of subset which has few parts occur
    int cnt = 0;
    for(int i = 0; i < subset.size(); i++) {
        if (subset[i][SUBSET_CNT]<1) {
            LOG(INFO) << "BAD SUBSET_CNT";
        }
        if (subset[i][SUBSET_CNT]>=connect_min_subset_cnt && (subset[i][SUBSET_SCORE]/subset[i][SUBSET_CNT])>connect_min_subset_score) {
            for(int j = 0; j < num_parts; j++) {
                int idx = int(subset[i][j]);
                if (idx) {
                    joints[cnt*num_parts*3 + j*3 +2] = peaks[idx];
                    joints[cnt*num_parts*3 + j*3 +1] = peaks[idx-1]* image_height/ (float)NET_RESOLUTION_HEIGHT;//(peaks[idx-1] - padh) * ratio_h;
                    joints[cnt*num_parts*3 + j*3] = peaks[idx-2]* image_width/ (float)NET_RESOLUTION_WIDTH;//(peaks[idx-2] -padw) * ratio_w;
                }
                else{
                    joints[cnt*num_parts*3 + j*3 +2] = 0;
                    joints[cnt*num_parts*3 + j*3 +1] = 0;
                    joints[cnt*num_parts*3 + j*3] = 0;
                }
            }
            cnt++;
            if (cnt==MAX_PEOPLE) break;
        }
    }

    return cnt;
}


void RTPose::render(float *heatmaps /*GPU*/, int image_width, int image_height, int part_to_show, int googly_eyes) {
    float* centers = 0;
    float* poses    = netcopy.joints;

    double tic = get_wall_time();
    if (netcopy.up_model_descriptor->get_number_parts()==15) {
        render_mpi_parts(netcopy.canvas, image_width, image_height, NET_RESOLUTION_WIDTH, NET_RESOLUTION_HEIGHT,
        heatmaps, BOX_SIZE, centers, poses, netcopy.num_people, part_to_show);
    } else if (netcopy.up_model_descriptor->get_number_parts()==18) {
        if (part_to_show-1<=netcopy.up_model_descriptor->get_number_parts()) {
            render_coco_parts(netcopy.canvas,
            image_width, image_height,
            NET_RESOLUTION_WIDTH, NET_RESOLUTION_HEIGHT,
            heatmaps, BOX_SIZE, centers, poses,
            netcopy.num_people, part_to_show, googly_eyes);
        } else {
            int aff_part = ((part_to_show-1)-netcopy.up_model_descriptor->get_number_parts()-1)*2;
            int num_parts_accum = 1;
            if (aff_part==0) {
                num_parts_accum = 19;
            } else {
                aff_part = aff_part-2;
                }
                aff_part += 1+netcopy.up_model_descriptor->get_number_parts();
                render_coco_aff(netcopy.canvas, image_width, image_height, NET_RESOLUTION_WIDTH, NET_RESOLUTION_HEIGHT,
                heatmaps, BOX_SIZE, centers, poses, netcopy.num_people, aff_part, num_parts_accum);
        }
    }
    VLOG(2) << "Render time " << (get_wall_time()-tic)*1000.0 << " ms.";
}

Frame RTPose::postProcessFrame(Frame frame, int image_width, int image_height) {

    frame.postprocesse_begin_time = get_wall_time();

    //Mat visualize(NET_RESOLUTION_HEIGHT, NET_RESOLUTION_WIDTH, CV_8UC3);
    int offset = image_width * image_height;
    for(int c = 0; c < 3; c++) {
        for(int i = 0; i < image_height; i++) {
            for(int j = 0; j < image_width; j++) {
                int value = int(frame.data_for_mat[c*offset + i*image_width + j] + 0.5);
                value = value<0 ? 0 : (value > 255 ? 255 : value);
                frame.data_for_wrap[3*(i*image_width + j) + c] = (unsigned char)(value);
            }
        }
    }
    frame.postprocesse_end_time = get_wall_time();
    return frame;
}

std::vector<unsigned char> RTPose::run(std::vector<unsigned char> input, param_map & params) {
    cv::Mat data_mat(input, true);
    cv::Mat cvImageIn(imdecode(data_mat, CV_LOAD_IMAGE_COLOR)); //put 0 if you want greyscale
    
    int image_width = cvImageIn.cols;
    int image_height = cvImageIn.rows;
    int part_to_show_param = part_to_show;
    int googly_eyes_param = 0;
    if (params.count("googly_eyes")) {
        googly_eyes_param = std::stol(params["googly_eyes"]);
        if (googly_eyes_param != 0 && googly_eyes_param != 1) {
            LOG(INFO) << "googly_eyes param with invalid value " << googly_eyes_param << " ignored.";
            googly_eyes_param = 0;
        }
    }
    if (params.count("part_to_show")) {
        part_to_show_param = std::stol(params["part_to_show"]);
        if (part_to_show_param < 0 || part_to_show_param > 55) {
            LOG(INFO) << "part_to_show param with invalid value " << part_to_show_param << " ignored.";
            part_to_show_param = part_to_show;
        }
    }
    if (image_width > MAX_RESOLUTION_WIDTH || image_height > MAX_RESOLUTION_HEIGHT) {
        LOG(INFO) << "Image too large: " << image_width << "x" << image_height << ". Ignoring.";
        std::vector<unsigned char> result;
        cv::putText(cvImageIn, 
            "Input image larger than " + std::to_string(MAX_RESOLUTION_WIDTH) + "x" + std::to_string(MAX_RESOLUTION_HEIGHT),
             cv::Point(100, image_height/2), cv::FONT_HERSHEY_PLAIN, 3, cv::Scalar(147, 20, 255), 2 ,8);
        imencode(".jpg", cvImageIn, result);
        return result;
    }

    // this is unnecessary copying of frames...
    Frame inputFrame = getFrame(cvImageIn, image_width, image_height);
    Frame processed = processFrame(inputFrame, image_width, image_height, part_to_show_param, googly_eyes_param);
    Frame postProcessed = postProcessFrame(processed, image_width, image_height);

    cv::Mat cvImageOut(image_height, image_width, CV_8UC3, postProcessed.data_for_wrap);

    delete [] postProcessed.data_for_wrap;
    delete [] postProcessed.data;
    delete [] postProcessed.data_for_mat;

    std::vector<unsigned char> result;
    std::vector<int> p;
    //p.push_back(CV_IMWRITE_JPEG_QUALITY);
    //p.push_back(10);
    imencode(".jpg", cvImageOut, result, p);
    return result;
}

Frame RTPose::getFrame(cv::Mat input, int image_width, int image_height) {
    double scale = 0;
    cv::Mat image_uchar;
    if (input.cols/(double)input.rows>image_width/(double)image_height) {
        scale = image_width/(double)input.cols;
    } else {
        scale = image_height/(double)input.rows;
    }
    cv::Mat M = cv::Mat::eye(2,3,CV_64F);
    M.at<double>(0,0) = scale;
    M.at<double>(1,1) = scale;
    cv::warpAffine(input, image_uchar, M,
                         cv::Size(image_width, image_height),
                         CV_INTER_CUBIC,
                         cv::BORDER_CONSTANT, cv::Scalar(0,0,0));
    if ( image_uchar.empty() ) std::cout << "empty" << std::endl;

    Frame frame;
    frame.ori_width = input.cols;
    frame.ori_height = input.rows;
    frame.index = global_counter++;
    frame.video_frame_number = frame.index;
    frame.data_for_wrap = new unsigned char [image_height * image_width * 3]; //fill after process
    frame.data_for_mat = new float [image_height * image_width * 3];
    process_and_pad_image(frame.data_for_mat, image_uchar, image_width, image_height, 0);

    frame.scale = scale;
    //pad and transform to float
    int offset = 3 * NET_RESOLUTION_HEIGHT * NET_RESOLUTION_WIDTH;
    frame.data = new float [BATCH_SIZE * offset];
    int target_width, target_height;
    cv::Mat image_temp;
    //LOG(ERROR) << "frame.index: " << frame.index;
    for(int i=0; i < BATCH_SIZE; i++) {
        float scale = START_SCALE - i*SCALE_GAP;
        target_width = 16 * ceil(NET_RESOLUTION_WIDTH * scale /16);
        target_height = 16 * ceil(NET_RESOLUTION_HEIGHT * scale /16);

        CHECK_LE(target_width, NET_RESOLUTION_WIDTH);
        CHECK_LE(target_height, NET_RESOLUTION_HEIGHT);

        resize(image_uchar, image_temp, cv::Size(target_width, target_height), 0, 0, CV_INTER_AREA);
        process_and_pad_image(frame.data + i * offset, image_temp, NET_RESOLUTION_WIDTH, NET_RESOLUTION_HEIGHT, 1);
    }
    frame.commit_time = get_wall_time();
    frame.preprocessed_time = get_wall_time();
    return frame;
}

Frame RTPose::processFrame(Frame frame, int image_width, int image_height, int part_to_show, int googly_eyes) {

    int offset = NET_RESOLUTION_WIDTH * NET_RESOLUTION_HEIGHT * 3;
    //bool empty = false;

    // TODO: is this needed
    Frame frame_batch;

    std::vector< std::vector<double>> subset;
    std::vector< std::vector< std::vector<double> > > connection;

    const boost::shared_ptr<caffe::Blob<float>> heatmap_blob = netcopy.person_net->blob_by_name("resized_map");
    const boost::shared_ptr<caffe::Blob<float>> joints_blob = netcopy.person_net->blob_by_name("joints");

    caffe::NmsLayer<float> *nms_layer = (caffe::NmsLayer<float>*)netcopy.person_net->layer_by_name("nms").get();

    frame.gpu_fetched_time = get_wall_time();
    
    cudaMemcpy(netcopy.canvas, frame.data_for_mat, image_width * image_height * 3 * sizeof(float), cudaMemcpyHostToDevice);

    frame_batch = frame;
    //LOG(ERROR)<< "Copy data " << index_array[n] << " to device " << tid << ", now size " << global.input_queue.size();
    float* pointer = netcopy.person_net->blobs()[0]->mutable_gpu_data();

    cudaMemcpy(pointer + 0 * offset, frame_batch.data, BATCH_SIZE * offset * sizeof(float), cudaMemcpyHostToDevice);

    nms_layer->SetThreshold(nms_threshold);
    netcopy.person_net->ForwardFrom(0);

    VLOG(2) << "CNN time " << (get_wall_time()-frame.gpu_fetched_time)*1000.0 << " ms.";
    //cudaDeviceSynchronize();
    float* heatmap_pointer = heatmap_blob->mutable_cpu_data();
    const float* peaks = joints_blob->mutable_cpu_data();

    float joints[MAX_NUM_PARTS*3*MAX_PEOPLE]; //10*15*3

    int cnt = 0;

    // CHECK_EQ(net_copies[tid].nms_num_parts, 15);
    double tic = get_wall_time();
    const int num_parts = netcopy.nms_num_parts;

    if (netcopy.nms_num_parts==15) {
        cnt = connectLimbs(subset, connection,
                                             heatmap_pointer, peaks,
                                             netcopy.nms_max_peaks, joints, netcopy.up_model_descriptor.get(),
                                             image_width, image_height);
    } else {
        cnt = connectLimbsCOCO(subset, connection,
                                             heatmap_pointer, peaks,
                                             netcopy.nms_max_peaks, joints, netcopy.up_model_descriptor.get(),
                                             image_width, image_height);
    }

    VLOG(2) << "CNT: " << cnt << " Connect time " << (get_wall_time()-tic)*1000.0 << " ms.";
    netcopy.num_people[0] = cnt;
    VLOG(2) << "num_people[i] = " << cnt;

    cudaMemcpy(netcopy.joints, joints,
        MAX_NUM_PARTS*3*MAX_PEOPLE * sizeof(float),
        cudaMemcpyHostToDevice);

    if (subset.size() != 0) {
        //LOG(ERROR) << "Rendering";
        render(heatmap_pointer, image_width, image_height, part_to_show, googly_eyes); //only support batch size = 1!!!!
        frame_batch.numPeople = netcopy.num_people[0];
        frame_batch.gpu_computed_time = get_wall_time();
        frame_batch.joints = boost::shared_ptr<float[]>(new float[frame_batch.numPeople*MAX_NUM_PARTS*3]);
        for (int ij=0;ij<frame_batch.numPeople*num_parts*3;ij++) {
            frame_batch.joints[ij] = joints[ij];
        }


        cudaMemcpy(frame_batch.data_for_mat, netcopy.canvas, image_height * image_width * 3 * sizeof(float), cudaMemcpyDeviceToHost);
        return frame_batch;
    }
    else {
        render(heatmap_pointer, image_width, image_height, part_to_show, googly_eyes);
        //frame_batch[n].data should revert to 0-255
        frame_batch.numPeople = 0;
        frame_batch.gpu_computed_time = get_wall_time();
        cudaMemcpy(frame_batch.data_for_mat, netcopy    .canvas, image_height * image_width * 3 * sizeof(float), cudaMemcpyDeviceToHost);
        return frame_batch;
    }
}



RTPose::RTPose(int deviceId) {
    //google::InitGoogleLogging("rtcpm");
    warmup(deviceId);
}

void RTPose::warmup(int device_id) {
    int logtostderr = FLAGS_logtostderr;

    LOG(INFO) << "Setting GPU " << device_id;

    caffe::Caffe::SetDevice(device_id); //cudaSetDevice(device_id) inside
    caffe::Caffe::set_mode(caffe::Caffe::GPU); //

    LOG(INFO) << "GPU " << device_id << ": copying to person net";
    FLAGS_logtostderr = 0;
    netcopy.person_net = new caffe::Net<float>(PERSON_DETECTOR_PROTO, caffe::TEST);
    netcopy.person_net->CopyTrainedLayersFrom(PERSON_DETECTOR_CAFFEMODEL);

    netcopy.nblob_person = netcopy.person_net->blob_names().size();
    netcopy.num_people.resize(BATCH_SIZE);
    const std::vector<int> shape { {BATCH_SIZE, 3, NET_RESOLUTION_HEIGHT, NET_RESOLUTION_WIDTH} };

    netcopy.person_net->blobs()[0]->Reshape(shape);
    netcopy.person_net->Reshape();
    FLAGS_logtostderr = logtostderr;

    caffe::NmsLayer<float> *nms_layer = (caffe::NmsLayer<float>*)netcopy.person_net->layer_by_name("nms").get();
    netcopy.nms_max_peaks = nms_layer->GetMaxPeaks();


    caffe::ImResizeLayer<float> *resize_layer =
        (caffe::ImResizeLayer<float>*)netcopy.person_net->layer_by_name("resize").get();

    resize_layer->SetStartScale(START_SCALE);
    resize_layer->SetScaleGap(SCALE_GAP);
    LOG(INFO) << "start_scale = " << START_SCALE;

    netcopy.nms_max_peaks = nms_layer->GetMaxPeaks();

    netcopy.nms_num_parts = nms_layer->GetNumParts();
    CHECK_LE(netcopy.nms_num_parts, MAX_NUM_PARTS)
        << "num_parts in NMS layer (" << netcopy.nms_num_parts << ") "
        << "too big ( MAX_NUM_PARTS )";

    if (netcopy.nms_num_parts==15) {
        ModelDescriptorFactory::createModelDescriptor(ModelDescriptorFactory::Type::MPI_15, netcopy.up_model_descriptor);
        nms_threshold = 0.2;
        connect_min_subset_cnt = 3;
        connect_min_subset_score = 0.4;
        connect_inter_threshold = 0.01;
        connect_inter_min_above_threshold = 8;
        LOG(INFO) << "Selecting MPI model.";
    } else if (netcopy.nms_num_parts==18) {
        ModelDescriptorFactory::createModelDescriptor(ModelDescriptorFactory::Type::COCO_18, netcopy.up_model_descriptor);
        nms_threshold = 0.05;
        connect_min_subset_cnt = 3;
        connect_min_subset_score = 0.4;
        connect_inter_threshold = 0.050;
        connect_inter_min_above_threshold = 9;
    } else {
        CHECK(0) << "Unknown number of parts! Couldn't set model";
    }

    //dry run
    LOG(INFO) << "Dry running...";
    netcopy.person_net->ForwardFrom(0);
    LOG(INFO) << "Success.";
    cudaMalloc(&netcopy.canvas, MAX_RESOLUTION_WIDTH * MAX_RESOLUTION_HEIGHT * 3 * sizeof(float));
    cudaMalloc(&netcopy.joints, MAX_NUM_PARTS*3*MAX_PEOPLE * sizeof(float) );
}