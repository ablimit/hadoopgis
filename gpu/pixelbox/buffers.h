#ifndef BUFFERS_H
#define BUFFERS_H

#include <pthread.h>
#include "spatial.h"
#include "constants.h"
#include "spatialindex.h"


//#define file_names_input	file_names_all_images
//#define file_names_input	file_names_astroII_1
//#define file_names_input	file_names_astroII_2
//#define file_names_input	file_names_gbm0_1
//#define file_names_input	file_names_gbm0_2
//#define file_names_input	file_names_gbm1_1
//#define file_names_input	file_names_gbm1_2
//#define file_names_input	file_names_gbm2_1
//#define file_names_input	file_names_gbm2_2
//#define file_names_input	file_names_normal_2
//#define file_names_input	file_names_normal_3
//#define file_names_input	file_names_oligoastroII_1
//#define file_names_input	file_names_oligoastroII_2
//#define file_names_input	file_names_oligoastroIII_1
//#define file_names_input	file_names_oligoastroIII_2
//#define file_names_input	file_names_oligoII_1
//#define file_names_input	file_names_oligoII_2
//#define file_names_input	file_names_oligoIII_1
#define file_names_input	file_names_oligoIII_2

// the buffer item in the work buffer
struct file_names_buffer_item
{
	char *file_name_1;
	char *file_name_2;
};

struct poly_arrays_buffer_item
{
	poly_array_t *polys_1;
	poly_array_t *polys_2;
};

struct spatial_data_buffer_item
{
	poly_array_t *polys_1;
	spatial_index_t *index_1;
	poly_array_t *polys_2;
	spatial_index_t *index_2;
};

struct poly_pairs_buffer_item
{
	poly_pair_array_t *poly_pairs;
	poly_array_t *polys_1;
	poly_array_t *polys_2;
public:
	inline int size()
	{
		return poly_pairs->nr_poly_pairs;
	}
};


template <class BufferItem>
class dequeue
{
public:
	BufferItem			*items;
	const int			queue_size;
	int					head;	// this is where the next task to be poped
	int					tail;	// this is where the next task to be pushed
	bool				exiting;
	pthread_mutex_t		rw_lock;
	char pad1[64];
	pthread_cond_t		wl_pullers;
	char pad2[64];
	pthread_cond_t		wl_pushers;
	char pad3[64];
	pthread_cond_t		wl_diverter;

public:
	dequeue(const int size = 1):
		head(0), tail(0), queue_size(size+1), exiting(false)
	{
		items = new BufferItem[size+1];
		pthread_mutex_init(&rw_lock, NULL);
		pthread_cond_init(&wl_pullers, NULL);
		pthread_cond_init(&wl_pushers, NULL);
		pthread_cond_init(&wl_diverter, NULL);
	}

	~dequeue()
	{
		delete items;
	}

	// get the index of an item to be poped
	inline int pop_index()
	{
		int ret;

		if(head == tail) {
			ret = -1;
		}
		else {
			ret = head;
			head = (head + 1) % queue_size;
		}

		return ret;
	}

	// get an index for the item to be pushed
	inline int push_index()
	{
		int ret;

		if((tail + 1) % queue_size == head)
			ret = -1;
		else {
			ret = tail;
			tail = (tail + 1) % queue_size;
		}

		return ret;
	}

	inline int peek_load()
	{
		return (tail - head + queue_size) % queue_size;
	}

	inline bool is_over_loading()
	{
		return peek_load() > (queue_size - 1) * 4 / 5;
	}

	inline bool is_under_loading()
	{
		return peek_load() < (queue_size - 1) / 2;
	}

	inline bool is_empty()
	{
		return (head == tail);
	}

	inline int lock()
	{
		return pthread_mutex_lock(&rw_lock);
	}

	inline int try_lock()
	{
		return pthread_mutex_trylock(&rw_lock);
	}

	inline void unlock()
	{
		pthread_mutex_unlock(&rw_lock);
	}

	inline void wait_for_slot()
	{
		pthread_cond_wait(&wl_pushers, &rw_lock);
	}

	inline void wake_up_pusher()
	{
		pthread_cond_signal(&wl_pushers);
	}

	inline void wait_for_task()
	{
		pthread_cond_wait(&wl_pullers, &rw_lock);
	}

	inline void wake_up_puller()
	{
		pthread_cond_signal(&wl_pullers);
	}

	inline void wake_up_diverter()
	{
		pthread_cond_signal(&wl_diverter);
	}

	inline void wait_for_congestion()
	{
		pthread_cond_wait(&wl_diverter, &rw_lock);
	}

	void signal_exit()
	{
		lock();
		exiting = true;
		wake_up_puller();	// wake up other sleeping pullers
		wake_up_diverter();	// wake up the diverter thread
		unlock();
	}

	int pull_task(BufferItem *item, pthread_cond_t *cond = NULL)
	{
		int ret = 0, popidx;

		lock();

	repeat:
		popidx = pop_index();
		if(popidx < 0) {
			if(exiting) {
				ret = -1;
				wake_up_puller();
			}
			else {
				if(cond)
					pthread_cond_signal(cond);
				wait_for_task();
				wake_up_puller();	// wake up other possible pullers
				goto repeat;
			}
		}
		else {
			*item = items[popidx];
			if(peek_load() == queue_size - 2) {
				wake_up_pusher();
			}
		}

		unlock();

		return ret;
	}

	// TODO: pull the smallest task
	void pull_task_nolock(BufferItem *item)
	{
		int i_min = head, size_min = 1000000;

		for(int i = head; i != tail; i = (i+1) % queue_size) {
			if(items[i].size() < size_min) {
				size_min = items[i].size();
				i_min = i;
			}
		}

		if(i_min != head) {
			*item = items[i_min];
			items[i_min] = items[head];
			head = (head + 1) % queue_size;
		}
		else {
			*item = items[head];
			head = (head + 1) % queue_size;
		}

		if(peek_load() == queue_size - 2)
			wake_up_pusher();

/*		int popidx = pop_index();

		*item = items[popidx];
		if(peek_load() == queue_size - 2)
			wake_up_pusher();*/
	}

	int push_task(BufferItem *item)
	{
		int pushidx;

		lock();

	repeat:
		pushidx = push_index();
		if(pushidx < 0) {
			wait_for_slot();
			wake_up_pusher();	// wake up other possible pushers
			goto repeat;
		}
		else {
			items[pushidx] = *item;
			if(peek_load() == 1) {
				wake_up_puller();
			}
			if(is_over_loading()) {
				wake_up_diverter();
			}
		}

		unlock();

		return 0;
	}
};

// the work buffer for storing file names
class file_names_buffer	: public dequeue<file_names_buffer_item>
{
public:
	file_names_buffer(const int size = 1) :
		dequeue<file_names_buffer_item>(size) {}

	int pull_task(file_names_buffer_item *item)
	{
		int ret = 0;

		lock();

		if(file_names_input[head][0]) {
			item->file_name_1 = file_names_input[head][0];
			item->file_name_2 = file_names_input[head][1];
			head++;
		}
		else {
			ret = -1;	// signal the end of pipelined processing
		}

		unlock();

		return ret;
	}

	int pull_task_nolock(file_names_buffer_item *item)
	{
		int ret = 0;

		if(file_names_input[head][0]) {
			item->file_name_1 = file_names_input[head][0];
			item->file_name_2 = file_names_input[head][1];
			head++;
		}
		else {
			ret = -1;	// signal the end of pipelined processing
		}

		return ret;
	}
};

#endif
