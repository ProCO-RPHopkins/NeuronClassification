using Random, CSV, DataFrames

# Function to generate synthetic data
function generate_data(n:: Int)
    data = DataFrame(id = 1:n, firing_rate = rand(50:150, n), class = rand(["Excitatory", "Inhibitory"], n))
    CSV.write("neuron_data.csv", data)

end

# Generate data
generate_data(1000)

# Load the data and display the first 5 rows
using CSV, DataFrames

data = CSV.read("neuron_data.csv", DataFrame)
println(first(data, 5))

# Explore the data
println(describe(data))

# Data processing
using MLJ

data.class = coerce(data.class, Multiclass)

# Split the data
using MLJBase

train, test = partition(eachindex(data.id), 0.8, shuffle=true)
train_data = data[train, :]
test_data = data[test, :]

# Define the machine learning model
using Flux
using Optimisers # Import optimisers for ADAM

model = Chain(
    Dense(1, 10, relu),
    Dense(10, 2),
    softmax
)

# Training the model - prepare data for training
X_train = Matrix(train_data[:, :firing_rate]')
y_train = Flux.onehotbatch(train_data.class, ["Excitatory", "Inhibitory"])

# Train the model
loss(model, x, y) = Flux.crossentropy(model(x), y)
opt = Flux.setup(Optimisers.ADAM(), model)

# Training loop
train_data_tuple = [(X_train, y_train)]
parameters = Flux.params(model)
for epoch in 1:100
    Flux.train!(loss, params(model), train_data_tuple, opt)

end

# Print the trained model
println("Trained Model:")
println(model)

# Evaluating the model - prepare data for testing
X_test = Matrix(test_data[:, :firing_rate]')
y_test = test_data.class

# Evaluating the model - make predictions
y_pred = model(X_test)
y_pred_labels = Flux.onecold(y_pred, ["Excitatory", "Inhibitory"])

# Evaluating the model - calculate accuracy
accuracy = mean(y_pred_labels .== y_test)
println("Model Accuracy: ", accuracy)

# Unable to run the plot because of permissions restrictions!!!
# Visualize the data
using Plots
pyplot()

# Plot the data
scatter(data.id, data.firing_rate, group=data.class, legend=:top, title="Neuron Firing Rates", xlabel="Neuron ID", ylabel="Firing Rate")

# Save the first plot
savefig("neuron_firing_rates.png")

# Add predictions to the test data
test_data.predicted_class = y_pred_labels

# Plot the actual vs predicted classifications
p = scatter(test_data.id, test_data.firing_rate, group=test_data.class, legend=:topright, title="Actual vs Predicted Classifications", xlabel="Neuron ID", ylabel="Firing Rate", label="Actual")
scatter!(p, test_data.id, test_data.firing_rate, group=test_data.predicted_class, markershape=:cross, label="Predicted")

# Save the second plot
savefig(p, "actual_vs_predicted.png")

# Display the plots
display(p)