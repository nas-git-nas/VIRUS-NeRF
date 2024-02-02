



def main():
    data_dir = "results/pso/opt32_2"
    plotter = PlotterResults(
        data_dir=data_dir,
    )
    plotter.plot()


if __name__ == "__main__":
    main()