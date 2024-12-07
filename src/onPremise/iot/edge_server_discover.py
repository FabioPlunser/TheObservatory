import logging
import asyncio
from zeroconf import ServiceBrowser, ServiceListener, Zeroconf
from zeroconf.asyncio import AsyncZeroconf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EdgeServerDiscovery(ServiceListener):
    def __init__(self):
        self.edge_server_url = None
        self._discovery_event = asyncio.Event()
        
    def update_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        logger.info(f"Service {name} updated")
        
    def remove_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        logger.info(f"Service {name} removed")
        
    def add_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        info = zc.get_service_info(type_, name)
        if info:
            addresses = info.parsed_addresses()
            if addresses:
                ip = addresses[0]
                port = info.port
                self.edge_server_url = f"http://{ip}:{port}"
                logger.info(f"Found edge server at {self.edge_server_url}")
                self._discovery_event.set()

    async def discover_edge_server(self):
        """Discover edge server asynchronously"""
        try:
            aiozc = AsyncZeroconf()
            browser = ServiceBrowser(aiozc.zeroconf, "_edgeserver._tcp.local.", self)
            
            # Wait for discovery or timeout
            try:
                await asyncio.wait_for(self._discovery_event.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("Discovery timed out")
            
            await aiozc.async_close()
            return self.edge_server_url
            
        except Exception as e:
            logger.error(f"Error discovering edge server: {e}")
            return None