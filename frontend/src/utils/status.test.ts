import { describe, it, expect } from 'vitest'
import {
  DEPLOYMENT_STATUS_COLORS,
  WORKER_STATUS_COLORS,
  CONTAINER_STATUS_COLORS,
  getDeploymentStatusColor,
  getWorkerStatusColor,
  getContainerStatusColor,
} from './status'

describe('Status Color Constants', () => {
  describe('DEPLOYMENT_STATUS_COLORS', () => {
    it('has correct color mappings', () => {
      expect(DEPLOYMENT_STATUS_COLORS.running).toBe('green')
      expect(DEPLOYMENT_STATUS_COLORS.pending).toBe('blue')
      expect(DEPLOYMENT_STATUS_COLORS.downloading).toBe('cyan')
      expect(DEPLOYMENT_STATUS_COLORS.starting).toBe('blue')
      expect(DEPLOYMENT_STATUS_COLORS.stopping).toBe('orange')
      expect(DEPLOYMENT_STATUS_COLORS.stopped).toBe('default')
      expect(DEPLOYMENT_STATUS_COLORS.error).toBe('red')
    })
  })

  describe('WORKER_STATUS_COLORS', () => {
    it('has correct color mappings', () => {
      expect(WORKER_STATUS_COLORS.online).toBe('green')
      expect(WORKER_STATUS_COLORS.offline).toBe('default')
      expect(WORKER_STATUS_COLORS.error).toBe('red')
    })
  })

  describe('CONTAINER_STATUS_COLORS', () => {
    it('has correct color mappings', () => {
      expect(CONTAINER_STATUS_COLORS.running).toBe('green')
      expect(CONTAINER_STATUS_COLORS.created).toBe('blue')
      expect(CONTAINER_STATUS_COLORS.restarting).toBe('orange')
      expect(CONTAINER_STATUS_COLORS.paused).toBe('orange')
      expect(CONTAINER_STATUS_COLORS.exited).toBe('default')
      expect(CONTAINER_STATUS_COLORS.dead).toBe('red')
    })
  })
})

describe('Status Color Functions', () => {
  describe('getDeploymentStatusColor', () => {
    it('returns correct color for known statuses', () => {
      expect(getDeploymentStatusColor('running')).toBe('green')
      expect(getDeploymentStatusColor('error')).toBe('red')
      expect(getDeploymentStatusColor('pending')).toBe('blue')
    })

    it('returns default for unknown status', () => {
      expect(getDeploymentStatusColor('unknown')).toBe('default')
      expect(getDeploymentStatusColor('')).toBe('default')
    })
  })

  describe('getWorkerStatusColor', () => {
    it('returns correct color for known statuses', () => {
      expect(getWorkerStatusColor('online')).toBe('green')
      expect(getWorkerStatusColor('offline')).toBe('default')
      expect(getWorkerStatusColor('error')).toBe('red')
    })

    it('returns default for unknown status', () => {
      expect(getWorkerStatusColor('unknown')).toBe('default')
    })
  })

  describe('getContainerStatusColor', () => {
    it('returns correct color for known statuses', () => {
      expect(getContainerStatusColor('running')).toBe('green')
      expect(getContainerStatusColor('dead')).toBe('red')
      expect(getContainerStatusColor('paused')).toBe('orange')
    })

    it('returns default for unknown status', () => {
      expect(getContainerStatusColor('unknown')).toBe('default')
    })
  })
})
