#!/usr/bin/env python3
"""
Reset Manager Module
Handles all reset-related operations
"""

import time
from rclpy.node import Node
from std_srvs.srv import Trigger
from edge_drl_student.common.logger import Logger


class ResetManager:
    """
    Manages reset operations with different triggers:
    - Episode done
    - Fall detection
    - Manual reset
    - Timeout
    """
    
    def __init__(self, 
                 node : Node, 
                 reset_service_name: str,
                 reset_timeout: float, 
                 logger: Logger,
                 auto_reset_on_done=True, 
                 auto_reset_on_fall=True):
        """
        Initialize Reset Manager
        
        Args:
            node: ROS2 node instance
            reset_service_name: Reset service name
            reset_timeout: Service call timeout
            logger: Logger instance
            auto_reset_on_done: Auto reset when episode done
            auto_reset_on_fall: Auto reset when fall detected
        """
        self.node = node
        self.reset_service_name = reset_service_name
        self.reset_timeout = reset_timeout
        self.logger = logger
        self.auto_reset_on_done = auto_reset_on_done
        self.auto_reset_on_fall = auto_reset_on_fall
        
        # Create reset client
        self.reset_client = node.create_client(Trigger, reset_service_name)
        
        # Statistics
        self.reset_count = 0
        self.reset_success_count = 0
        self.reset_failure_count = 0
        self.is_resetting = False
        
        # Reset reason statistics
        self.reset_reasons = {
            'episode_done': 0,
            'fall_detected': 0,
            'manual': 0,
            'timeout': 0
        }
        
        # Fall detection state
        self.last_fall_state = False
        
        self.node.get_logger().info(f"Reset Manager initialized: {reset_service_name}")
        self.node.get_logger().info(f"Auto reset on done: {auto_reset_on_done}, on fall: {auto_reset_on_fall}")
    
    def call_reset_service(self, reason='unknown'):
        """
        Call reset service
        
        Args:
            reason: Reset reason
            
        Returns:
            tuple: (success: bool, message: str)
        """
        if self.is_resetting:
            return False, "Reset already in progress"
        
        self.is_resetting = True
        self.reset_count += 1
        
        # Update reason statistics
        if reason in self.reset_reasons:
            self.reset_reasons[reason] += 1
        
        self.node.get_logger().info(
            f'[Reset #{self.reset_count}] Reason: {reason} | '
            f'Waiting for service: {self.reset_service_name}...'
        )
        
        reset_start = time.time()
        
        # Wait for service
        if not self.reset_client.wait_for_service(timeout_sec=self.reset_timeout):
            error_msg = f'Service not available after {self.reset_timeout}s'
            self.node.get_logger().error(error_msg)
            self.reset_failure_count += 1
            self.is_resetting = False
            self._log_reset_failure(reason)
            return False, error_msg
        
        # Create and send request
        request = Trigger.Request()
        
        try:
            future = self.reset_client.call_async(request)
            
            # Simple polling
            start_time = time.time()
            while not future.done():
                elapsed = time.time() - start_time
                
                if elapsed > self.reset_timeout:
                    error_msg = f'Service call timeout after {elapsed:.2f}s'
                    self.node.get_logger().error(error_msg)
                    self.reset_failure_count += 1
                    self._log_reset_failure(reason)
                    self.is_resetting = False
                    return False, error_msg
                
                time.sleep(0.01)
            
            elapsed_time = time.time() - reset_start
            
            # Get result
            response = future.result()
            
            if response is None:
                error_msg = f'Service returned None response'
                self.node.get_logger().error(error_msg)
                self.reset_failure_count += 1
                self._log_reset_failure(reason)
                self.is_resetting = False
                return False, error_msg
            
            if response.success:
                self.reset_success_count += 1
                self.node.get_logger().info(
                    f'✅ Reset successful (reason: {reason}) in {elapsed_time:.2f}s'
                )
                self._log_reset_success(elapsed_time, reason)
                self.is_resetting = False
                return True, response.message
            else:
                self.reset_failure_count += 1
                self.node.get_logger().warn(
                    f'❌ Reset failed (reason: {reason}): {response.message}'
                )
                self._log_reset_failure(reason)
                self.is_resetting = False
                return False, response.message
                
        except Exception as e:
            error_msg = f'Exception during service call: {type(e).__name__}: {str(e)}'
            self.node.get_logger().error(error_msg)
            self.reset_failure_count += 1
            self._log_reset_failure(reason)
            self.is_resetting = False
            return False, error_msg
    
    def _log_reset_success(self, elapsed_time, reason):
        """Log successful reset statistics"""
        self.logger.writer.add_scalar('Episode/Reset/success', 1, self.reset_count)
        self.logger.writer.add_scalar('Episode/Reset/time', elapsed_time, self.reset_count)
        self.logger.writer.add_scalar(f'Episode/Reset/reason_{reason}', 1, self.reset_count)
        
        success_rate = self.reset_success_count / self.reset_count if self.reset_count > 0 else 0.0
        self.logger.writer.add_scalar('Episode/Reset/success_rate', success_rate, self.reset_count)
        
        for r, count in self.reset_reasons.items():
            self.logger.writer.add_scalar(f'Episode/Reset/count_{r}', count, self.reset_count)
    
    def _log_reset_failure(self, reason):
        """Log failed reset statistics"""
        self.logger.writer.add_scalar('Episode/Reset/success', 0, self.reset_count)
        
        success_rate = self.reset_success_count / self.reset_count if self.reset_count > 0 else 0.0
        failure_rate = self.reset_failure_count / self.reset_count if self.reset_count > 0 else 0.0
        
        self.logger.writer.add_scalar('Episode/Reset/success_rate', success_rate, self.reset_count)
        self.logger.writer.add_scalar('Episode/Reset/failure_rate', failure_rate, self.reset_count)
    
    def handle_fall_detection(self, is_fall):
        """
        Handle fall detection callback
        
        Args:
            is_fall: Current fall state
            
        Returns:
            bool: True if reset was triggered
        """
        # Detect rising edge (transition from False to True)
        if is_fall and not self.last_fall_state:
            self.node.get_logger().warn('⚠️ Fall detected!')
            
            if self.auto_reset_on_fall:
                self.node.get_logger().info('🔄 Auto reset triggered by fall detection')
                self.last_fall_state = is_fall
                return True
        
        self.last_fall_state = is_fall
        return False
    
    def get_statistics(self):
        """Get reset statistics"""
        return {
            'total': self.reset_count,
            'success': self.reset_success_count,
            'failure': self.reset_failure_count,
            'reasons': self.reset_reasons.copy()
        }
