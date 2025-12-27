// ============================================================
// EchoZero Data Bus
// Connects oscillator nodes to central hub
// ============================================================

module echozero_bus #(
    parameter N_NODES = 64,
    parameter DATA_WIDTH = 32  // Complex: 16-bit real + 16-bit imag
)(
    input wire clk,
    input wire rst_n,

    // Node interfaces
    input wire [N_NODES*DATA_WIDTH-1:0] node_data_in,
    output reg [N_NODES*DATA_WIDTH-1:0] node_data_out,

    // Hub interface
    output reg [DATA_WIDTH-1:0] hub_data,
    input wire [DATA_WIDTH-1:0] hub_broadcast,

    // Control
    input wire enable,
    output reg valid
);

    // Internal registers
    reg [DATA_WIDTH-1:0] node_reg [0:N_NODES-1];

    integer i;

    // Bus arbiter
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (i = 0; i < N_NODES; i = i + 1) begin
                node_reg[i] <= 32'h0;
            end
            hub_data <= 32'h0;
            valid <= 1'b0;
        end else if (enable) begin
            // Gather from nodes
            for (i = 0; i < N_NODES; i = i + 1) begin
                node_reg[i] <= node_data_in[i*DATA_WIDTH +: DATA_WIDTH];
            end

            // Sum for hub (global coupling)
            hub_data <= sum_nodes();

            // Broadcast from hub
            for (i = 0; i < N_NODES; i = i + 1) begin
                node_data_out[i*DATA_WIDTH +: DATA_WIDTH] <= hub_broadcast;
            end

            valid <= 1'b1;
        end else begin
            valid <= 1'b0;
        end
    end

    // Function to sum nodes (simplified)
    function [DATA_WIDTH-1:0] sum_nodes;
        integer j;
        reg [DATA_WIDTH-1:0] sum;
        begin
            sum = 32'h0;
            for (j = 0; j < N_NODES; j = j + 1) begin
                sum = sum + node_reg[j];  // Simplified addition
            end
            sum_nodes = sum;
        end
    endfunction

endmodule
