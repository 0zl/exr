import asyncio
import aioredis
import json, time, random, gc, time
from aioredis.client import PubSub

import shared
from exllama import EXL

async def serialize(identifier, data, requestId=None, callbackId=None):
    timestamp = int(round(time.time() * 1000))
    return json.dumps({
        'ts': timestamp,
        'from': identifier,
        'data': data,
        'requestId': requestId,
        'callbackId': callbackId
    })


class RDSClient:
    client: aioredis.Redis
    pcl: PubSub
    exl: EXL

    def __init__(self, gs, exl: EXL):
        self.host = gs.rd_h
        self.port = gs.rd_p
        self.secret = gs.rd_s
        self.identifier = gs.rd_id
        self.mother = gs.rd_m
        self.global_chan = gs.rd_g
        self.ts_started = int(round(time.time() * 1000))
        self.exl = exl

    async def send_data(self, data, success=True, request_id=None, channel=None):
        if hasattr(data, '__dict__'):
            data = data.__dict__
        elif type(data) is list:
            list_data = []
            for i, e in enumerate(data):
                list_data.append(e)
            data = { 'list': list_data }
        elif isinstance(data, str):
            data = { 'info': data }

        n_data = { 'success': success, **data }
        serialized_data = await serialize(self.identifier, n_data, callbackId=request_id)
        
        await self.client.publish(
            self.mother if channel is None else channel,
            serialized_data
        )
    
    async def ping_status(self, channel=None):
        await self.send_data({
            'id': self.identifier,
            'ts': self.ts_started,
            'status': 'online' if shared.init_complete else 'booting',
            'busy_queue': shared.exr_busy
        }, channel=channel)
    
    async def generate(self, task, params, chan_name, request_id):
        if task == 'streaming':
            try:
                shared.exr_busy = True

                tensor_input, gen_settings, max_response_tokens = self.exl.prepare_stream(
                    params['prompt'],
                    params['stop_condition'] + [self.exl.tokenizer.eos_token_id],
                    params['temperature'],
                    params['top_k'],
                    params['top_p'],
                    params['typical'],
                    params['repitition_penalty'],
                    params['max_response_tokens']
                )

                response_tokens = 0
                self.exl.streaming.begin_stream(tensor_input, gen_settings)
                
                t_stream_start = time.time()
                while True:
                    chunk, eos, _ = self.exl.streaming.stream()
                    response_tokens += 1

                    await self.send_data({
                        'output': chunk,
                        'eos': False,
                        'stream_id': params.stream_id
                    }, True)

                    if eos or response_tokens == max_response_tokens:
                        await self.send_data({
                            'output': '',
                            'eos': True,
                            'stream_id': params.stream_id
                        }, True)
                        break
                t_stream_end = time.time()
                t_tokens = t_stream_end - t_stream_start

                print(f'stream process: {response_tokens / t_tokens:.2f} t/s')
            except Exception as e:
                await self.send_data(str(e), False, request_id, chan_name)
            finally:
                shared.exr_busy = False
                gc.collect()
        else:
            print(f'unknown task: {task}')

    async def handle_global_call(self, msg):
        if msg['channel'] != self.global_chan:
            return False

        await self.ping_status()
        return True
    
    async def handle_tasks(self, msg):
        data = json.loads(msg['data'])

        chan_name = data['from']
        request_id = data['requestId']

        try:
            task_name = data['data']['task']
            task_params = data['data']['params']
        except Exception as e:
            await self.send_data(str(e), False, request_id, chan_name)
            return
        
        print(task_params)
        if task_name in ['streaming']:
            await self.generate(task_name, task_params, chan_name, request_id)
        else:
            await self.ping_status(chan_name)

    async def process_message(self, msg):
        if msg['type'] not in ['subscribe', 'message']:
            return
        
        if not isinstance(msg['data'], str):
            print(msg)
            return
        
        if await self.handle_global_call(msg):
            return
        
        await self.handle_tasks(msg)
    
    async def _launch_async(self):
        self.client = aioredis.Redis(
            host=self.host,
            port=self.port,
            password=self.secret,
            decode_responses=True
        )
        
        self.pcl = self.client.pubsub()
        await self.pcl.subscribe(self.global_chan)
        await self.pcl.subscribe(self.identifier)

        async for msg in self.pcl.listen():
            asyncio.create_task(self.process_message(msg))
    
    def launch(self):
        print('evt loop - exl - rd')
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self._launch_async())
