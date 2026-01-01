import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { HeroUIProvider } from '@heroui/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import App from '../App';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: false,
    },
  },
});

const renderApp = () => {
  return render(
    <QueryClientProvider client={queryClient}>
      <HeroUIProvider>
        <App />
      </HeroUIProvider>
    </QueryClientProvider>
  );
};

describe('App', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders loading state initially', () => {
    renderApp();
    expect(screen.getByText(/Initializing Edge-Net/i)).toBeInTheDocument();
  });

  it('renders main dashboard after loading', async () => {
    renderApp();

    await waitFor(
      () => {
        expect(screen.getByText(/Network Overview/i)).toBeInTheDocument();
      },
      { timeout: 3000 }
    );
  });

  it('renders header with Edge-Net branding', async () => {
    renderApp();

    await waitFor(
      () => {
        expect(screen.getByText('Edge-Net')).toBeInTheDocument();
      },
      { timeout: 3000 }
    );
  });

  it('shows connection status', async () => {
    renderApp();

    await waitFor(
      () => {
        expect(screen.getByText(/Connected/i)).toBeInTheDocument();
      },
      { timeout: 3000 }
    );
  });
});
