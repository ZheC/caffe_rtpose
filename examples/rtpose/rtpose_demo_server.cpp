#include "rtpose/server_http.hpp"
#include <iostream>
#include <mimetic/mimetic.h>
#define BOOST_SPIRIT_THREADSAFE
#include <boost/asio.hpp>
#include <boost/thread.hpp>
#include <boost/bind.hpp>
//Added for the default_resource example
#include "rtpose/rtpose_demo.hpp"
#include <fstream>
#include <boost/filesystem.hpp>
#include <vector>
#include <algorithm>

using namespace std;
using namespace mimetic;

typedef SimpleWeb::Server<SimpleWeb::HTTP> HttpServer;
typedef std::unordered_multimap<std::string, std::string, case_insensitive_hash, case_insensitive_equals> query_params;

// CLI Flags
DEFINE_int32(port,          8080,              "Port to run webserver on.");
DEFINE_int32(gpu_device,    0,              "GPU to use (0 is first GPU).");

stringstream getImage(MimeEntity* pMe)
{
	MimeEntityList& parts = pMe->body().parts(); // list of sub entities obj
	// cycle on sub entities list and print info of every item
	MimeEntityList::iterator mbit = parts.begin(), meit = parts.end();
	stringstream image;
	for(; mbit != meit; ++mbit) {
		Header& h = (*mbit)->header(); // get header object
		if (h.contentDisposition().param("name") == "image") {
			string body;
			(*mbit)->body().load(body);
			image << (*mbit)->body();
		}
	}
	return image;
}



struct request_data {
  shared_ptr<HttpServer::Response> response;
  shared_ptr<HttpServer::Request> request;

  typedef std::shared_ptr<request_data> pointer;

  request_data(shared_ptr<HttpServer::Response> response, shared_ptr<HttpServer::Request> request)
      : response(response), request(request) {}
};

/**
 * A basic work queue
 */
struct work_queue {
  typedef std::list<request_data::pointer> list;

  list requests;
  boost::mutex mutex;

  inline void put(const request_data::pointer& p_rd) {
    boost::unique_lock<boost::mutex> lock(mutex);
    requests.push_back(p_rd);
    (void)lock;
  }

  inline request_data::pointer get() {
    boost::unique_lock<boost::mutex> lock(mutex);

    request_data::pointer p_ret;
    if (!requests.empty()) {
      p_ret = requests.front();
      requests.pop_front();
    }

    (void)lock;

    return p_ret;
  }
};

void process_request(work_queue& queue, int deviceId) {

  RTPose rtpose(deviceId);

  while (!boost::this_thread::interruption_requested()) {
    request_data::pointer p_req(queue.get());
    if (p_req) {
       	   	double start = get_wall_time();
    	    try {
          	string boundary;
            for(auto& header: p_req->request->header) {
            	if (header.first == "Content-Type") {
            		string split = "boundary=";
            		std::size_t found = header.second.find(split);
            		  if (found!=std::string::npos)
            		  	boundary = header.second;
 	       		}
			}
			if (!boundary.empty()) {
				    stringstream content_stream;
				    content_stream << "Content-Type: " << boundary << "\r\n\r\n";
				    content_stream << p_req->request->content.string();
				    content_stream.seekg(0, content_stream.beg);
				    istreambuf_iterator<char> bit(content_stream), eit;
				    MimeEntity me(bit, eit);
				    stringstream image = getImage(&me);
					image.seekg(0, ios::end);
					int size = image.tellg();
				    if (size == 0) {
				    	string content="Image parameter missing..."; 
 						*(p_req->response) << "HTTP/1.1 400 Bad Request\r\nContent-Length: " << content.length() << "\r\n\r\n" << content;              
				    }
				    else {
   						image.seekg(0, ios::beg);
				    	string img_data = image.str();
				    	std::vector<unsigned char> vectordata(img_data.begin(),img_data.end());
				    	auto result = rtpose.run(vectordata, p_req->request->query_string);
				    	std::string content(result.begin(), result.end());
				    	//string content = img_data;
		 				*(p_req->response) << "HTTP/1.1 200 OK\r\nContent-Length: " << content.length() << "\r\n"
		 				          << "Access-Control-Allow-Origin: *\r\n"
		 				          << "Content-Type: image/jpeg\r\n\r\n" << content;              
				    }

			} else {
 				string content="Only mulipart/form-data requests allowed..."; 
 				*(p_req->response) << "HTTP/1.1 400 Bad Request\r\nContent-Length: " << content.length() << "\r\n\r\n" << content;              
 				      //<< "Content-Type: application/json\r\n";
                      //<< "Content-Length: " << name.length() << "\r\n\r\n"
                      //<< name;
			}
        }
        catch(exception& e) {
            *(p_req->response) << "HTTP/1.1 400 Bad Request\r\nContent-Length: " << strlen(e.what()) << "\r\n\r\n" << e.what();
        }
        double end = get_wall_time();
        double elapsed = end - start;
        LOG(INFO) << "Done processing request from " << p_req->request->remote_endpoint_address << " (" << p_req->request->remote_endpoint_port << ") in " << elapsed;
    }
    boost::this_thread::sleep(boost::posix_time::microseconds(1000));
  }
}


int main(int argc, char *argv[]) {
    google::InitGoogleLogging(argv[0]);
    // Parsing command line flags
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    HttpServer server;
    server.config.port = FLAGS_port;

	std::shared_ptr<boost::thread_group> p_threads(std::make_shared<boost::thread_group>());
    work_queue queue;
    // worker threads that will process the request off the queue
	{
	  // currently only one thread with one GPU -- add additional here...
	  int n_threads = 1;
	  while (0 < n_threads--) {
	    p_threads->create_thread(boost::bind(process_request, boost::ref(queue), FLAGS_gpu_device));
	  }
	}

    server.resource["^/predict$"]["POST"]=[&server, &queue](shared_ptr<HttpServer::Response> response, shared_ptr<HttpServer::Request> request) {
        {
		   	LOG(INFO) << "Received request " << request->method << " " << request->path << " from " << request->remote_endpoint_address << " (" << request->remote_endpoint_port << ")" << std::endl;
      	    queue.put(std::make_shared<request_data>(response, request));
        };
    };
      
    thread server_thread([&server](){
        //Start server
        LOG(INFO) << "Starting to listen on port " << FLAGS_port;
        server.start();
    });

   
    server_thread.join();

    p_threads->interrupt_all();
    
    return 0;
}
