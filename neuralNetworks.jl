using Random, CSV, DataFrames

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
data = [(X_train, y_train)]
parameters = Flux.params(model)
for epoch in 1:100
    Flux.train!(loss, params(model), data, opt)
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
